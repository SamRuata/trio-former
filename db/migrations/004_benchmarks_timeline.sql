-- 004_benchmarks_timeline.sql
-- A time series for the interface tier's Latency History tab.
--
-- The `benchmarks` view answers "what does each model cost", which is one row
-- per model. Charting latency OVER TIME needs a time axis, and the obvious
-- shortcut -- letting the interface read `inferences` directly with the anon
-- key -- is exactly what this project's privacy argument forbids: that table
-- holds T5 summaries, which are lossy copies of what users typed.
--
-- So the interface gets aggregates and never rows. Hourly buckets, counts and
-- percentiles only; no output column is exposed, and no individual call can
-- be recovered from it.

drop materialized view if exists benchmarks_timeline;
create materialized view benchmarks_timeline as
select
    date_trunc('hour', served_at)                     as bucket,
    environment,
    model_name,
    model_id,
    archetype,
    count(*)                                          as n_calls,
    count(*) filter (where cold_start)                as n_cold_starts,
    percentile_cont(0.5) within group (order by latency_ms)::int
                                                      as median_ms,
    percentile_cont(0.5) within group (order by latency_ms)
        filter (where not cold_start)::int            as median_warm_ms,
    max(latency_ms)                                   as max_ms
from inferences
group by 1, 2, 3, 4, 5;

create unique index if not exists idx_timeline
    on benchmarks_timeline (bucket, environment, model_name, model_id);

grant select on benchmarks_timeline to anon;

-- Refresh both views together, so GET /benchmarks keeps the chart current too.
create or replace function refresh_benchmarks()
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
    begin
        refresh materialized view concurrently benchmarks;
    exception when others then
        refresh materialized view benchmarks;
    end;
    begin
        refresh materialized view concurrently benchmarks_timeline;
    exception when others then
        refresh materialized view benchmarks_timeline;
    end;
end;
$$;