-- 001_trioformer.sql
-- Data layer for TrioFormer: three transformer archetypes behind one service.
--
-- The template's schema was not applied. It models a product that trains its
-- own model and persists runs and artifacts; this product trains nothing and
-- serves three pretrained models, so `datasets`, `runs`, `run_artifacts` and
-- `sequence_metadata` would all sit empty. The cost story here lives in
-- latency per call, not in training metrics.

-- ---------------------------------------------------------------------------
-- inputs: one row per distinct prompt, identified by hash.
--
-- The prompt text itself is never written. The hash lets the same paragraph
-- submitted twice be recognised as the same paragraph -- which is what makes
-- a fair latency comparison possible, since the three models must be timed on
-- identical input -- without the database holding what anyone typed.
-- ---------------------------------------------------------------------------
create table if not exists inputs (
    id            bigint generated always as identity primary key,
    input_hash    text    not null unique,
    n_chars       integer not null,
    n_words       integer not null,
    first_seen_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- inferences: one row per model call.
--
-- Unlike `inputs`, this table DOES store the model's output. That asymmetry is
-- deliberate and is not privacy-neutral: a T5 summary is a lossy copy of the
-- paragraph it summarised, so hashing the input does not make the record
-- non-identifying. The README says so rather than implying the hash settles it.
-- ---------------------------------------------------------------------------
create table if not exists inferences (
    id          bigint generated always as identity primary key,
    input_id    bigint not null references inputs (id) on delete cascade,

    model_name  text    not null,   -- the endpoint: bert_fillmask, gpt2_continue, t5_summarize
    model_id    text    not null,   -- the checkpoint actually served, e.g. distilbert-base-uncased
    archetype   text    not null
                check (archetype in ('encoder', 'decoder', 'encoder-decoder')),
    task        text    not null,   -- fill-mask, text-generation, summarization

    output      jsonb   not null,
    latency_ms  integer not null,   -- measured server-side, around the forward pass only
    cold_start  boolean not null default false,  -- true when this call paid to load the model
    served_at   timestamptz not null default now()
);

create index if not exists idx_inferences_model   on inferences (model_name);
create index if not exists idx_inferences_served  on inferences (served_at desc);
create index if not exists idx_inferences_input   on inferences (input_id);

-- ---------------------------------------------------------------------------
-- benchmarks: the cost story, aggregated by model.
--
-- Cold starts are counted and excluded from the warm percentiles. On a free
-- 512 MB instance only one model stays resident, so every switch between
-- archetypes pays a load -- averaging that into the per-call latency would
-- describe the hosting tier rather than the model.
-- ---------------------------------------------------------------------------
drop materialized view if exists benchmarks;
create materialized view benchmarks as
select
    model_name,
    model_id,
    archetype,
    task,
    count(*)                                          as n_calls,
    count(*) filter (where cold_start)                as n_cold_starts,
    round(avg(latency_ms) filter (where not cold_start))::int  as mean_warm_ms,
    percentile_cont(0.5) within group (order by latency_ms)
        filter (where not cold_start)::int            as p50_warm_ms,
    percentile_cont(0.95) within group (order by latency_ms)
        filter (where not cold_start)::int            as p95_warm_ms,
    min(latency_ms) filter (where not cold_start)     as min_warm_ms,
    max(latency_ms) filter (where not cold_start)     as max_warm_ms,
    round(avg(latency_ms) filter (where cold_start))::int      as mean_cold_ms,
    max(served_at)                                    as last_served_at
from inferences
group by model_name, model_id, archetype, task;

create unique index if not exists idx_benchmarks_model
    on benchmarks (model_name, model_id);

-- ---------------------------------------------------------------------------
-- A materialized view does not update itself. GET /benchmarks calls this
-- first, so the numbers a caller sees include the call they just made.
-- CONCURRENTLY needs the unique index above and avoids locking readers out.
-- ---------------------------------------------------------------------------
create or replace function refresh_benchmarks()
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
    refresh materialized view concurrently benchmarks;
exception when others then
    -- CONCURRENTLY fails until the view has been populated once.
    refresh materialized view benchmarks;
end;
$$;

-- ---------------------------------------------------------------------------
-- Row Level Security. The API writes with the service-role key, which bypasses
-- RLS. No anonymous client reads these tables in this product -- the interface
-- tier is Product 5B and talks to the API, not to Postgres -- but read-only
-- anon access to the aggregate is granted so that it can, without ever
-- exposing a single call's output.
-- ---------------------------------------------------------------------------
alter table inputs     enable row level security;
alter table inferences enable row level security;

grant select on benchmarks to anon;