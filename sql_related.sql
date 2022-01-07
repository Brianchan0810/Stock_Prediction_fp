CREATE VIEW actual_prediction_joint AS
SELECT GOOGL_price.date, MONTH(GOOGL_price.date) AS month_of_date, YEAR(GOOGL_price.date) AS year_of_date, (CLOSE > OPEN) AS movement, prediction
FROM GOOGL_pred
LEFT JOIN GOOGL_price
ON  GOOGL_pred.date = GOOGL_price.date;


CREATE VIEW semi_model_performance AS
SELECT COUNT(1) AS total_rows, 
SUM(CASE WHEN movement = prediction THEN 1 ELSE 0 END) AS correct,
SUM(CASE WHEN movement = 1 THEN 1 ELSE 0 END) AS real_pos, 
SUM(CASE WHEN movement = 1 AND prediction = 1 THEN 1 ELSE 0 END) AS true_pos,
SUM(CASE WHEN movement = 0 AND prediction = 1 THEN 1 ELSE 0 END) AS false_pos
FROM actual_prediction_joint;


CREATE VIEW model_performance AS
SELECT (correct / total_rows) AS model_accuracy, (true_pos / (true_pos + false_pos)) AS model_precision, (true_pos / real_pos) AS model_recall 
FROM semi_model_performance;


SELECT GOOGL_be_sent.date, GOOGL_be_sent.compound_mean AS premarket_sent, new_GOOGL_af_sent.compound_mean AS ytd_postmarket_sent
FROM GOOGL_be_sent
LEFT JOIN 
(SELECT *, LEAD(DATE, 1) OVER (ORDER BY DATE) AS next_date FROM GOOGL_af_sent) AS new_GOOGL_af_sent
ON new_GOOGL_af_sent.next_date = GOOGL_be_sent.date;

SELECT * FROM GOOGL_be_sent;

CREATE VIEW monthly_model_performance AS 
SELECT year_of_date, month_of_date, SUM(CASE WHEN movement = prediction THEN 1 ELSE 0 END) /COUNT(1) AS monthly_accuracy
FROM actual_prediction_joint
GROUP BY year_of_date, month_of_date;


SELECT *, (CASE WHEN DATE<CURRENT_DATE THEN prediction ELSE 'not ready' END) AS display FROM
(SELECT *, (CASE WHEN prediction=1 THEN 'goes up' ELSE 'goes down' END) AS pred_str,  FROM GOOGL_pred
ORDER BY DATE DESC
LIMIT 5) AS a;


SELECT t1.CURRENT_DATE, (CASE WHEN t2.prediction=1 THEN 'goes up'  WHEN t2.prediction=0 THEN 'goes down' ELSE 'not ready' END) FROM
(SELECT CURRENT_DATE) AS t1
LEFT JOIN GOOGL_pred AS t2
ON t1.CURRENT_DATE = t2.date







