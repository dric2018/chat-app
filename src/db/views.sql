CREATE OR REPLACE VIEW vw_winners AS
SELECT
    r.REGION_NAME,
    c.CANDIDATE_NAME,
    circ.CONSTITUENCY_NUM,
    p.PARTY_NAME,
    res.SCORES,
    res.PCT_SCORE
FROM result res
JOIN candidate c ON res.CANDIDATE_ID = c.CANDIDATE_ID
JOIN party p ON res.PARTY_ID = p.PARTY_ID
JOIN constituency circ ON c.CONSTITUENCY_NUM = circ.CONSTITUENCY_NUM
JOIN region r ON circ.REGION_ID = r.REGION_ID
WHERE res.IS_WINNER = TRUE; 

--- "How many seats did Party X win in region Y?"
CREATE OR REPLACE VIEW vw_party AS
SELECT 
    p.PARTY_NAME,
    r.REGION_NAME,
    SUM(res.SCORES) AS TOTAL_VOTES,
    -- Ensure this counts across the whole group
    COUNT(CASE WHEN res.IS_WINNER = TRUE THEN 1 END) AS SEATS_WON
FROM result res
JOIN candidate cand ON res.CANDIDATE_ID = cand.CANDIDATE_ID
JOIN party p ON cand.PARTY_ID = p.PARTY_ID
JOIN constituency circ ON res.CONSTITUENCY_ID = circ.CONSTITUENCY_ID
JOIN region r ON circ.REGION_ID = r.REGION_ID
-- Explicitly group by the high-level entities only
GROUP BY p.PARTY_NAME, r.REGION_NAME;

CREATE OR REPLACE VIEW vw_turnout AS
SELECT 
    t.TURNOUT_ID,
    r.REGION_NAME,
    c.CONSTITUENCY_NUM,
    t.NUM_POLLING_STATIONS,
    t.REGISTERED,
    t.PART_RATE,
    t.NUM_VOTERS,
    t.EXPRESSED_VOTES,
    t.NUM_BLANK,
    t.NULL_BALL,
    t.PCT_BLANK,
FROM turnout t
JOIN constituency c ON t.CONSTITUENCY_NUM = c.CONSTITUENCY_NUM
JOIN region r ON c.REGION_ID = r.REGION_ID;

CREATE OR REPLACE VIEW vw_results AS
SELECT 
    res.RESULT_ID,
    r.REGION_NAME,
    c.CONSTITUENCY_NUM,
    c.CONSTITUENCY_TITLE,
    cand.CANDIDATE_NAME,
    p.PARTY_NAME,
    res.SCORES,
    res.PCT_SCORE,
    res.IS_WINNER,
FROM result res
JOIN candidate cand ON res.CANDIDATE_ID = cand.CANDIDATE_ID
JOIN party p ON cand.PARTY_ID = p.PARTY_ID
JOIN constituency c ON res.CONSTITUENCY_NUM = c.CONSTITUENCY_NUM
JOIN region r ON c.REGION_ID = r.REGION_ID;

--- pre-formating data into descriptive sentences for RAG
CREATE OR REPLACE VIEW vw_rag_descriptions AS
WITH winner_info AS (
    SELECT 
        CONSTITUENCY_NUM, 
        CANDIDATE_NAME || ' (' || PARTY_NAME || ')' AS WINNER_NAME
    FROM vw_results
    WHERE IS_WINNER = TRUE
)
--- Constituency Summaries
SELECT 
    'In constituency ' || r.CONSTITUENCY_NUM || ' (' || r.CONSTITUENCY_TITLE || '), ' ||
    ANY_VALUE(w.WINNER_NAME) || ' won the election. Full results: ' ||
    STRING_AGG(r.CANDIDATE_NAME || ' received ' || r.SCORES || ' votes', '; ' ORDER BY r.SCORES DESC) 
    AS TEXT_CHUNK,
    'constituency_summary' AS ENTITY_TYPE,
    r.CONSTITUENCY_NUM AS ENTITY_ID
FROM vw_results r
LEFT JOIN winner_info w ON r.CONSTITUENCY_NUM = w.CONSTITUENCY_NUM
GROUP BY r.CONSTITUENCY_NUM, r.CONSTITUENCY_TITLE

UNION ALL

--- Individual Candidate Results
SELECT 
    'Candidate (or candidate group) ' || CANDIDATE_NAME || ' representing ' || PARTY_NAME || 
    ' ran in ' || CONSTITUENCY_TITLE || '. They received ' || SCORES || 
    ' votes, finishing in rank ' || DENSE_RANK() OVER (PARTITION BY CONSTITUENCY_NUM ORDER BY SCORES DESC) || 
    '. Outcome: ' || (CASE WHEN is_winner THEN 'ELECTED' ELSE 'NOT ELECTED' END) || '.' AS TEXT_CHUNK,
    'candidate_result' AS ENTITY_TYPE,
    RESULT_ID AS ENTITY_ID
FROM vw_results

UNION ALL

--- Turnout Data
SELECT 
    'The voter turnout in constituency ' || CONSTITUENCY_NUM || ' (' || REGION_NAME || ') was ' || 
    ROUND(PART_RATE, 2) || '%. There were ' || REGISTERED || 
    ' registered voters and ' || EXPRESSED_VOTES || ' expressed votes including ' || NUM_BLANK || ' blank votes (' || PCT_BLANK || '%).' AS TEXT_CHUNK,
    'turnout' AS ENTITY_TYPE,
    TURNOUT_ID AS ENTITY_ID
FROM vw_turnout

UNION ALL

--- Party Performance
SELECT 
    'The party ' || PARTY_NAME || ' won a total of ' || SEATS_WON || 
    ' seats across ' || REGION_NAME || '. They received ' || TOTAL_VOTES || ' total votes in this region.' AS TEXT_CHUNK,
    'party_performance' AS ENTITY_TYPE,
    -1 AS ENTITY_ID
FROM vw_party

UNION ALL

--- Regional Summary
SELECT DISTINCT
    'In the region of ' || REGION_NAME || ', the winningest party was ' || 
    FIRST_VALUE(PARTY_NAME) OVER (PARTITION BY REGION_NAME ORDER BY SEATS_WON DESC) || 
    ' with ' || MAX(SEATS_WON) OVER (PARTITION BY REGION_NAME) || ' seats.' AS TEXT_CHUNK,
    'regional_summary' AS ENTITY_TYPE,
    -1 AS ENTITY_ID
FROM vw_party;
