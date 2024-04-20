function [params_chain, acceptance_chain] = runMCMC(allData, params_initial, n_iterations, step_size)
  % Allocate space for the parameter chain and acceptance history
  n_params = length(params_initial);
  params_chain = zeros(n_iterations, n_params);
  acceptance_chain = false(n_iterations, 1);

  % Initialize the chain with the initial parameters
  params_chain(1, :) = params_initial;
  current_logLik = logLikelihood(allData, params_initial);
  current_logPrior = priorFunction(params_initial);

  % MCMC loop
  for i = 2:n_iterations
    % Propose new parameters by perturbing the current ones with a normal distribution
    params_proposal = params_chain(i-1, :) + normrnd(0, step_size, [1, n_params]);
    proposal_logLik = logLikelihood(allData, params_proposal);
    proposal_logPrior = priorFunction(params_proposal);

    % Compute acceptance probability using the Metropolis-Hastings criterion
    % If proposal_logPrior is -Inf (proposal out of bounds), acceptance_prob will be 0
    acceptance_prob = exp(proposal_logLik + proposal_logPrior - current_logLik - current_logPrior);

    % Accept or reject the proposal based on the acceptance probability
    if rand() <= acceptance_prob
      params_chain(i, :) = params_proposal;  % Accept the proposal
      current_logLik = proposal_logLik;  % Update current log-likelihood
      current_logPrior = proposal_logPrior;  % Update current log-prior
      acceptance_chain(i) = true;  % Mark this iteration as accepted
    else
      params_chain(i, :) = params_chain(i-1, :);  % Reject the proposal, keep the current parameters
      % Note: acceptance_chain(i) remains false
    end
    if mod(i, 500) == 0
      fprintf('Iterations - [%d/%d]\n', i, n_iterations);
    end
  end
  fprintf('Completed MCMC Iterations - [%d/%d]\n', n_iterations, n_iterations);
end