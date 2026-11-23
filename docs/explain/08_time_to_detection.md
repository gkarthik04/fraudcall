# Early detection
**Measured as.** Pick the score threshold at which only 5 % of normal *validation* calls ever raise an alarm at any step. Then, on test calls, time-to-detection is the number of seconds until a fraud call's score first reaches that threshold. Calls that are never flagged count as missed.
**Why.** A fraud alert after the money has moved is useless, so how early it fires matters as much as final accuracy.
