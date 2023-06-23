### qhackathon-2023

## qhackathon 2023

## CODE REVIEW

Result: Problem Function to Find tau, N value

Initialize the optimizer with starting values of theta anf phi
```
while not converged do
  Run the Optimizer for a fixed number of iterations or until convergence
  Obtain the optimized values of tau and N
  Calculate the loss function using the optimized values
  if the new loss is lower than the previous loss then
    accept the new values of theta and phi
  else
    reject the new values and keep the previous values
  end
  return the optimized values of tau and N
end
```
