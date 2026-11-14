-- 003_environment.sql
-- Separate where a call ran from what ran.
--
-- The benchmarks view grouped by model alone, so the same checkpoint measured
-- on a laptop and on a 0.1 CPU instance landed in one row: a mean of 1861 ms
-- over a p50 of 37 ms and a p95 of 5618 ms, which describes neither machine.

truncate inferences, inputs restart identity cascade;

alter table inferences add column if not exists environment text not null default 'local';

drop materialized view if exists benchmarks;
create materialized view benchmarks as
select
    environment,
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
group by environment, model_name, model_id, archetype, task;

create unique index if not exists idx_benchmarks_model
    on benchmarks (environment, model_name, model_id);

grant select on benchmarks to anon;