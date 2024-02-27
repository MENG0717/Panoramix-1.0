# Development notes

## Random sampling

We set out to explore the parameter space for i. chemical composition of raw materials, ii. masses of raw materials in each binder mix (being a sample), and iii. the curing conditions. There are different ways to do the sampling procedure, while each comes with particular challenges. We tried the following methods:

1. *Random sampling from uniform distributions and then normalising* to 100% for i., and to 100 g for ii. This approach proved not viable, because the normalisation would on the one hand alter the intended bounds of the probability density function and, on the other hand, the likelihood of a sample falling into the desired binder classification (mostly CEM mixes for now) was extremely low. 
2. *Performing a grid search across the parameter space*. This approach poses significant computational challenges, because the number of possible combinations is extremely large (billions of combinations). Although we were able to solve the initial computational restrictions, it would be next to impossible to  explore the full parameter space because the number of parameters in i. to iii. would be multiplied with one another. However, It may be possible to draw random combinations out of the total space of possibilities.
3. Different forms of *consecutive sampling* could be feasible. The algorithm would randomly choose one variable, draw a random number from its range / PDF, randomly select the next variable, correct it's upper bound so that the random number cannot exceed the total value of 100% / 100g, and so on. However, there are a few problems with this approach as well. For instance, the last number in the sequence is practically pre-determined as it has to complete the total of 100. Or, the other way around, if the upper bound is below the remaining gap to fill this will not work either.
4. On [stackoverflow](https://stackoverflow.com/a/51450665/2075003) somebody provided a function using the Dirichlet distribution. However, the solution also uses some form of scaling, which alters the distribution again, i.e. makes it non-uniform (see [my comment](https://stackoverflow.com/questions/51448275/generating-a-list-of-random-numbers-using-custom-bounds-and-summing-to-a-desire#comment91230607_51450665) for a plot).

### Approaches that could be explored

- [ ] Create the full or partial (e.g., CEM I binder mixes only) parameter space by means of a grid search and then randomly choose combinations out of the full space.
- [x] https://stackoverflow.com/a/51450665/2075003
