-- 002_load_ms.sql
-- Separate the cost of loading a model from the cost of running it.
--
-- The first version timed only the forward pass, so a call that spent four
-- seconds pulling weights into memory reported the 120 ms it spent thinking.
-- On a one-slot registry that load is the dominant cost of switching
-- archetypes, and hiding it would have made the benchmarks describe the
-- hosting tier rather than the models.

alter table inferences add column if not exists load_ms integer default 0;

drop materialized view if exists benchmarks;
create materialized view benchmarks as
select
    model_name,
    model_id,
    archetype,
    task,
    count(*)                                          as n_calls,
    count(*) filter (where cold_start)                as n_cold_starts,
    round(avg(latency_ms) filter (where not cold_start))::int as mean_warm_ms,
    percentile_cont(0.5) within group (order by latency_ms)
        filter (where not cold_start)::int            as p50_warm_ms,
    percentile_cont(0.95) within group (order by latency_ms)
        filter (where not cold_start)::int            as p95_warm_ms,
    min(latency_ms) filter (where not cold_start)     as min_warm_ms,
    max(latency_ms) filter (where not cold_start)     as max_warm_ms,
    round(avg(latency_ms) filter (where cold_start))::int     as mean_cold_ms,
    round(avg(load_ms)    filter (where cold_start))::int     as mean_load_ms,
    max(served_at)                                    as last_served_at
from inferences
group by model_name, model_id, archetype, task;

create unique index if not exists idx_benchmarks_model
    on benchmarks (model_name, model_id);

grant select on benchmarks to anon;