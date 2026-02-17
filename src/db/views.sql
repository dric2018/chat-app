CREATE OR REPLACE VIEW vw_winners AS
SELECT
    r.region_name,
    c.candidate_name,
    p.party_name,
    f.votes,
    f.vote_percent
FROM fact_results f
JOIN dim_candidate c USING(candidate_id)
JOIN dim_party p USING(party_id)
JOIN dim_region r USING(region_id)
WHERE f.is_winner = TRUE;

CREATE OR REPLACE VIEW vw_party_seats AS
SELECT
    p.party_name,
    COUNT(*) AS seats
FROM fact_results f
JOIN dim_candidate c USING(candidate_id)
JOIN dim_party p USING(party_id)
WHERE f.is_winner = TRUE
GROUP BY p.party_name;

CREATE OR REPLACE VIEW vw_turnout AS
SELECT
    r.region_name,
    t.participation_rate,
    t.registered,
    t.voters
FROM fact_turnout t
JOIN dim_region r USING(region_id);
