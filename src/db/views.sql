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
    COUNT(CASE WHEN res.IS_WINNER THEN 1 END) AS SEATS_WON
FROM result res
JOIN candidate cand ON res.CANDIDATE_ID = cand.CANDIDATE_ID
JOIN party p ON cand.PARTY_ID = p.PARTY_ID
JOIN constituency circ ON res.CONSTITUENCY_ID = circ.CONSTITUENCY_ID
JOIN region r ON circ.REGION_ID = r.REGION_ID
GROUP BY ALL;


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
JOIN constituency c ON t.CONSTITUENCY_ID = c.CONSTITUENCY_ID
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
JOIN constituency c ON res.CONSTITUENCY_ID = c.CONSTITUENCY_ID
JOIN region r ON c.REGION_ID = r.REGION_ID;

--- pre-formating data into descriptive sentences for RAG
CREATE OR REPLACE VIEW vw_rag_descriptions AS
SELECT 
    'In constituency ' || CONSTITUENCY_NUM || ' (' || CONSTITUENCY_TITLE || ') of region ' ||
    REGION_NAME || ', ' || CANDIDATE_NAME || ' (' || PARTY_NAME  || ') received ' || SCORES || ' votes (' || ROUND(PCT_SCORE, 2) || '%)' ||
    (CASE WHEN is_winner THEN ' and was elected.' ELSE ' but has lost.' END) AS TEXT_CHUNK,
    'result' AS ENTITY_TYPE,
    RESULT_ID AS ENTITY_ID
FROM vw_results

UNION ALL

SELECT 
    'The voter turnout in constituency ' || CONSTITUENCY_NUM || ' (' || REGION_NAME || ') was ' || 
    ROUND(PART_RATE, 2) || '%. There were ' || REGISTERED || 
    ' registered voters and ' || EXPRESSED_VOTES || ' expressed votes including ' || NUM_BLANK || ' blank votes (' || PCT_BLANK || '%).' AS TEXT_CHUNK,
    'turnout' AS ENTITY_TYPE,
    TURNOUT_ID AS ENTITY_ID
FROM vw_turnout;