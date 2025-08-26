"""RAG example with pydantic-ai — using vector search to augment a chat agent.

Run pgvector with:

    mkdir postgres-data
    docker run --rm -e POSTGRES_PASSWORD=postgres \
        -p 54320:5432 \
        -v `pwd`/postgres-data:/var/lib/postgresql/data \
        pgvector/pgvector:pg17

Build the search DB with:

    uv run -m pydantic_ai_examples.rag build

Ask the agent a question with:

    uv run -m pydantic_ai_examples.rag search "How do I configure logfire to work with FastAPI?"
"""

from __future__ import annotations as _annotations

import asyncio
import re
import sys
import time
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
import httpx
import logfire
import pydantic_core
from google import genai
from google.genai.types import EmbedContentConfig
from pydantic import TypeAdapter
from typing_extensions import AsyncGenerator

from pydantic_ai import RunContext
from pydantic_ai.agent import Agent

from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider


# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_asyncpg()
logfire.instrument_pydantic_ai()


class RateLimiter:
    """Rate limiter to ensure we don't exceed 30 requests per minute for embeddings."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect the rate limit before allowing a request."""
        async with self.lock:
            now = time.time()
            # Remove requests outside the current window
            self.requests = [req_time for req_time in self.requests
                             if now - req_time < self.window_seconds]
            
            # If we're at the limit, wait until the oldest request expires
            if len(self.requests) >= self.max_requests:
                oldest_request = self.requests[0]
                wait_time = self.window_seconds - (now - oldest_request) + 0.1
                if wait_time > 0:
                    logfire.info(f'Rate limit reached, waiting {wait_time:.1f} seconds')
                    await asyncio.sleep(wait_time)
                    # After waiting, clean up old requests again
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests 
                                     if now - req_time < self.window_seconds]
            
            # Record this request
            self.requests.append(now)


# Create a global rate limiter for embedding requests (30 requests per minute)
embedding_rate_limiter = RateLimiter(max_requests=200, window_seconds=60)


@dataclass
class Deps:
    genai_client: genai.Client
    pool: asyncpg.Pool


provider = GoogleProvider(vertexai=True,
                          project='stromasys-projects',
                          location='us-east5')
model = GoogleModel('gemini-2.5-flash', provider=provider)
agent = Agent(model=model, deps_type=Deps)


@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    # Apply rate limiting before making the embedding request
    await embedding_rate_limiter.acquire()
    
    with logfire.span(
        'create embedding for {search_query=}', search_query=search_query
    ):
        response = await context.deps.genai_client.aio.models.embed_content(
            model='gemini-embedding-001',
            contents=[search_query],
            config=EmbedContentConfig(
                task_type='RETRIEVAL_QUERY',
                output_dimensionality=1536,  # Match the DB schema dimension
            ),
        )

    assert len(response.embeddings) == 1, (
        f'Expected 1 embedding, got {len(response.embeddings)}, doc query: {search_query!r}'
    )
    embedding = response.embeddings[0].values
    embedding_json = pydantic_core.to_json(embedding).decode()
    rows = await context.deps.pool.fetch(
        'SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8',
        embedding_json,
    )
    return '\n\n'.join(
        f'# {row["title"]}\nDocumentation URL:{row["url"]}\n\n{row["content"]}\n'
        for row in rows
    )


async def run_agent(question: str):
    """Entry point to run the agent and perform RAG based question answering."""
    genai_client = genai.Client()
    # Note: logfire.instrument_openai is not applicable for genai client

    logfire.info('Asking "{question}"', question=question)

    async with database_connect(False) as pool:
        deps = Deps(genai_client=genai_client, pool=pool)
        answer = await agent.run(question, deps=deps)
    print(answer.output)


#######################################################
# The rest of this file is dedicated to preparing the #
# search database, and some utilities.                #
#######################################################

# JSON document from
# https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992
DOCS_JSON = (
    'https://gist.githubusercontent.com/'
    'samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992/raw/'
    '80c5925c42f1442c24963aaf5eb1a324d47afe95/logfire_docs.json'
)


async def build_search_db():
    """Build the search database."""
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    sections = sessions_ta.validate_json(response.content)

    async with database_connect(True) as pool:
        with logfire.span('create schema'):
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)
        async with asyncio.TaskGroup() as tg:
            genai_client = genai.Client()
            for section in sections:
                tg.create_task(insert_doc_section(sem, genai_client, pool, section))


async def insert_doc_section(
    sem: asyncio.Semaphore,
    genai_client: genai.Client,
    pool: asyncpg.Pool,
    section: DocsSection,
) -> None:
    async with sem:
        url = section.url()
        exists = await pool.fetchval('SELECT 1 FROM doc_sections WHERE url = $1', url)
        if exists:
            logfire.info('Skipping {url=}', url=url)
            return

        # Apply rate limiting before making the embedding request
        await embedding_rate_limiter.acquire()

        with logfire.span('create embedding for {url=}', url=url):
            response = await genai_client.aio.models.embed_content(
                model='gemini-embedding-001',
                contents=[section.embedding_content()],
                config=EmbedContentConfig(
                    task_type='RETRIEVAL_DOCUMENT',
                    output_dimensionality=1536,  # Match the DB schema dimension
                ),
            )
        assert len(response.embeddings) == 1, (
            f'Expected 1 embedding, got {len(response.embeddings)}, doc section: {section}'
        )
        embedding = response.embeddings[0].values
        embedding_json = pydantic_core.to_json(embedding).decode()
        await pool.execute(
            'INSERT INTO doc_sections (url, title, content, embedding) VALUES ($1, $2, $3, $4)',
            url,
            section.title,
            section.content,
            embedding_json,
        )


@dataclass
class DocsSection:
    id: int
    parent: int | None
    path: str
    level: int
    title: str
    content: str

    def url(self) -> str:
        url_path = re.sub(r'\.md$', '', self.path)
        return (
            f'https://logfire.pydantic.dev/docs/{url_path}/#{slugify(self.title, "-")}'
        )

    def embedding_content(self) -> str:
        return '\n\n'.join((f'path: {self.path}', f'title: {self.title}', self.content))


sessions_ta = TypeAdapter(list[DocsSection])


# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(
    create_db: bool = False,
) -> AsyncGenerator[asyncpg.Pool, None]:
    server_dsn, database = (
        'postgresql://ai:ai@localhost:5432',
        'pydantic_ai_rag',
    )
    if create_db:
        with logfire.span('check and create DB'):
            conn = await asyncpg.connect(server_dsn)
            try:
                db_exists = await conn.fetchval(
                    'SELECT 1 FROM pg_database WHERE datname = $1', database
                )
                if not db_exists:
                    await conn.execute(f'CREATE DATABASE {database}')
            finally:
                await conn.close()

    pool = await asyncpg.create_pool(f'{server_dsn}/{database}')
    try:
        yield pool
    finally:
        await pool.close()


DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_sections (
    id serial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    title text NOT NULL,
    content text NOT NULL,
    embedding vector(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding ON doc_sections USING hnsw (embedding vector_l2_ops);
"""


def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """Slugify a string, to make it URL friendly."""
    # Taken unchanged from https://github.com/Python-Markdown/markdown/blob/3.7/markdown/extensions/toc.py#L38
    if not unicode:
        # Replace Extended Latin characters with ASCII, i.e. `žlutý` => `zluty`
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(rf'[{separator}\s]+', separator, value)


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else None
    if action == 'build':
        asyncio.run(build_search_db())
    elif action == 'search':
        if len(sys.argv) == 3:
            q = sys.argv[2]
        else:
            q = 'How do I configure logfire to work with FastAPI?'
        asyncio.run(run_agent(q))
    else:
        print(
            'uv run --extra examples -m pydantic_ai_examples.rag build|search',
            file=sys.stderr,
        )
        sys.exit(1)
