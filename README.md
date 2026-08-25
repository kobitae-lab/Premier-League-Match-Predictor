SHORT SUMMARY:

Purpose of project (to answer): Can you predict the outcome of a soccer match based on a teams recent performance?

Data obtained from: https://www.football-data.co.uk/englandm.php

Used Pandas for data manipulation.

Kept these data columns:
- Match date,
- Home Team,
- Away Team,
- Goals scored from both teams,
- Shots taken from both teams,
- Match result (win, draw, loss)


Used the average stats of the teams last 10 games

MODEL 1:
- only utilized 1 season of data, model could have got lucky guesses, skewing the results
- used logistic regression, 55% accuracy

MODEL 2:
- utilized previous 5 seasons of premier league data
- slightly worse accuracy but the results are more meaningful and trustworthy
- swapped goals and shots to the teams differential because that is what matters
- used logistic regression, 48% accuracy

MODEL 3:
- tested with a more complex model to find non-linear relationships, 51% accuracy


Conclusion:
- All models tested yielded similar results around 50% which is at least better than randomly guessing for this 3 class-classification problem
- small relationship but there are many other factors in a soccer game
- all models had difficulty predicting draws as teams would need to be very even
- don't bet on a team just because they have been doing 'bad' or 'good' recently
