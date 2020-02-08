SELECT
	f.origin,
	f.dest,
	max(p.seats) AS max_seats,
	round(count(*) / count(DISTINCT f.year), 0) AS flights_per_year,
	round(avg(f.distance), 2) AS avg_distance,
	round(avg(p.seats) / count(DISTINCT f.year), 2) AS avg_seats_per_year,
	round(avg(f.arr_delay), 2) AS avg_arr_delay
FROM fly.flights f
INNER JOIN fly.planes p
	ON f.tailnum = p.tailnum
GROUP BY origin, dest
HAVING
	avg_distance BETWEEN 300 AND 400
	AND flights_per_year > 5000
ORDER BY max_seats DESC
LIMIT 2;