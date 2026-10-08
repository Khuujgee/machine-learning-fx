-- Hourly news tone per currency from GDELT 2.0 (global news events), Oct 2023 -> today.
-- Run in Google BigQuery (free sandbox is enough). Check "This query will process ..." first:
-- it should be well under the free 1 TB/month. Then: Save results -> CSV (Google Drive), download it and
-- put it in data/gdelt/ inside the project.
--
-- hour      : UTC hour the articles were added (GDELT processes news within ~15 minutes)
-- tone      : article-weighted average tone (negative = negative coverage), roughly -10..+10
-- goldstein : average "conflict (-10) .. cooperation (+10)" score of the events
-- events / articles : how much coverage there was (spikes = something is happening)
WITH ev AS (
  SELECT
    TIMESTAMP_TRUNC(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATEADDED AS STRING)), HOUR) AS hour,
    cc, NumArticles, AvgTone, GoldsteinScale
  FROM `gdelt-bq.gdeltv2.events`,
       UNNEST([Actor1CountryCode, Actor2CountryCode]) AS cc
  WHERE DATEADDED >= 20231001000000
    AND cc IS NOT NULL
),
mapped AS (
  SELECT
    hour,
    CASE
      WHEN cc = 'USA' THEN 'USD'
      WHEN cc IN ('DEU','FRA','ITA','ESP','NLD','BEL','AUT','IRL','PRT','FIN','GRC',
                  'SVK','SVN','LUX','EST','LVA','LTU','HRV','CYP','MLT','EUR') THEN 'EUR'
      WHEN cc = 'GBR' THEN 'GBP'
      WHEN cc = 'JPN' THEN 'JPY'
      WHEN cc = 'CHE' THEN 'CHF'
      WHEN cc = 'AUS' THEN 'AUD'
      WHEN cc = 'CAN' THEN 'CAD'
      WHEN cc = 'NZL' THEN 'NZD'
      WHEN cc = 'SWE' THEN 'SEK'
      WHEN cc = 'NOR' THEN 'NOK'
      WHEN cc = 'DNK' THEN 'DKK'
      WHEN cc = 'POL' THEN 'PLN'
      WHEN cc = 'HUN' THEN 'HUF'
      WHEN cc = 'CZE' THEN 'CZK'
      WHEN cc = 'TUR' THEN 'TRY'
      WHEN cc = 'ZAF' THEN 'ZAR'
      WHEN cc = 'MEX' THEN 'MXN'
      WHEN cc = 'BRA' THEN 'BRL'
      WHEN cc = 'CHN' THEN 'CNH'
      WHEN cc = 'IND' THEN 'INR'
      WHEN cc = 'THA' THEN 'THB'
      WHEN cc = 'ISR' THEN 'ILS'
      WHEN cc = 'SGP' THEN 'SGD'
      WHEN cc = 'HKG' THEN 'HKD'
    END AS currency,
    NumArticles, AvgTone, GoldsteinScale
  FROM ev
)
SELECT
  hour,
  currency,
  COUNT(*)                                                         AS events,
  SUM(NumArticles)                                                 AS articles,
  ROUND(SAFE_DIVIDE(SUM(AvgTone * NumArticles), SUM(NumArticles)), 3) AS tone,
  ROUND(AVG(GoldsteinScale), 3)                                    AS goldstein
FROM mapped
WHERE currency IS NOT NULL
GROUP BY hour, currency
ORDER BY hour, currency
