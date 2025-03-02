clc;
clear;
warning('off', 'all');

%% Problem Definition
nVar = 2;        % Number of Unknown (Decision) Variables
VarSize = [1 nVar];   % Matrix Size of Decision Variables
VarMin_original = [0.01 0.01];
VarMax_original = [3500 100];  % Decision Variables Upper Bound

VarMin = VarMin_original; 
VarMax = VarMax_original;

%% Parameters of PSO
MaxIt = 100;   % Maximum Number of Iterations
nPop = 20;     % Population Size (Swarm Size)
wmin = 0.4;    % Minimum Inertia Weight
wmax = 0.9;    % Maximum Inertia Weight
c1 = 2;        % Personal Acceleration Coefficient
c2 = 2;        % Social Acceleration Coefficient

%% Parameters of Bat Algorithm
alpha = 0.9;     % Amplitude coefficient for Bat Algorithm
r0 = 0.5;        % Initial pulse rate
gamma = 0.9;     % Pulse rate decay factor
Fmax = 2;        % Maximum frequency
Fmin = 0;        % Minimum frequency
epsilon = 1e-6;  % Small constant for stability
beta = 1.5;        % Frequency update parameter

%% Initialization
% The Particle Template
empty_particle.Position = [];
empty_particle.Velocity = [];
empty_particle.Cost = [];
empty_particle.Best.Position = [];
empty_particle.Best.Cost = [];

empty_bestparticles.Position = [];
empty_bestparticles.Velocity = [];
empty_bestparticles.Cost = [];
empty_bestparticles.Best.Position = [];
empty_bestparticles.Best.Cost = [];

empty_chaoticParticle.Position = [];

% Create Population Array for PSO
particle = repmat(empty_particle, nPop, 1);
bestparticles = repmat(empty_particle, nPop / 5, 1);
chaoticParticle  = repmat(empty_chaoticParticle, nPop, 1);


% Initialize Global Best
GlobalBest.Cost = -inf;

% Initialize Population Members
for i = 1:nPop
    % Generate Random Solution for PSO
    particle(i).Position = unifrnd(VarMin, VarMax, VarSize);
    % Initialize Velocity
    particle(i).Velocity = zeros(VarSize);
    % Evaluation
    particle(i).Cost = optimizeSVMParkinsonHyperparams(particle(i).Position);
    % Update Personal Best
    particle(i).Best.Position = particle(i).Position;
    particle(i).Best.Cost = particle(i).Cost;
    % Update Global Best
    if particle(i).Best.Cost > GlobalBest.Cost
        GlobalBest = particle(i).Best;
    end
end


v = zeros(nPop, nVar);                    % Velocities
A = 0.5 * ones(1, nPop);                  % Amplitudes
r = 0.5 * ones(1, nPop);                  % Pulse rates
fitness = zeros(nPop, 1);
for i = 1:nPop
	x(i, :) = unifrnd(VarMin, VarMax, VarSize);
	fitness(i) = optimizeSVMParkinsonHyperparams(x(i,:));
end

if max(fitness) > GlobalBest.Cost
	[fitnessMax, index] = max(fitness);
	xgBest = x(index,:);
	xgBestCost = fitnessMax;
	GlobalBest.Position = xgBest;
	GlobalBest.Cost = xgBestCost;
end

alpha2 = 0.3;   % Coefficient acting on the adaptive equation
epsilon2 = 1e-1;

% Store Best Cost
BestCosts = zeros(MaxIt, 1);

%% Main Loop of Hybrid ICPSO and Improved Bat Algorithm
for it = 1:MaxIt
	
    % PSO Section
    for i = 1:nPop
		positions = arrayfun(@(x) x.Position, particle, 'UniformOutput', false);
		positions_matrix = cell2mat(positions'); % Convert positions to matrix format
		sigma_max = max(std(positions_matrix, 0, 1));
		distances = arrayfun(@(x) norm(x.Position - GlobalBest.Position), particle);
		d_max = max(distances);   
        % Calculate Average Distance and Population Standard Deviation
        d_avg = mean(arrayfun(@(x) norm(x.Position - GlobalBest.Position), particle));
		sigma_pop = std([particle.Position]);
        % Adaptive Inertia Weight Calculation
        if particle(i).Cost <= mean([particle.Cost])
            w = wmin + (wmax - wmin) * (1 - ((sigma_pop + alpha2 * d_avg) / (sigma_max + alpha2 * d_max)));
        else
            w = wmax;
        end

        % Update Velocity and Position
        particle(i).Velocity = w * particle(i).Velocity + c1 * rand(VarSize) .* (particle(i).Best.Position - particle(i).Position) + c2 * rand(VarSize) .* (GlobalBest.Position - particle(i).Position);
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        % Apply Boundaries
        particle(i).Position = max(particle(i).Position, VarMin);
        particle(i).Position = min(particle(i).Position, VarMax);
        
        % Evaluate
        particle(i).Cost = optimizeSVMParkinsonHyperparams(particle(i).Position);
        % Update Personal and Global Bests
        if particle(i).Cost > particle(i).Best.Cost
            particle(i).Best.Position = particle(i).Position;
            particle(i).Best.Cost = particle(i).Cost;
        end
		if particle(i).Best.Cost > GlobalBest.Cost
                GlobalBest = particle(i).Best;
        end
    end
	
	% Chaotic Update (Chaos-based improvement)
    [~, SortOrder] = sort([particle.Cost], 'descend');
    
    for j = 1:nPop / 5
        bestparticles(j) = particle(SortOrder(j));
    end
	
	for t = 1:nPop
		chaoticParticle(t).Position(1) = (particle(t).Position(1) - VarMin(1)) / (VarMax(1) - VarMin(1)); 
		chaoticParticle(t).Position(2) = (particle(t).Position(2) - VarMin(2)) / (VarMax(2) - VarMin(2)); 
		chaoticParticle(t).Position(1) = 4 * chaoticParticle(t).Position(1) * (1 - chaoticParticle(t).Position(1));
		chaoticParticle(t).Position(2) = 4 * chaoticParticle(t).Position(2) * (1 - chaoticParticle(t).Position(2));
		particle(t).Position(1) = VarMin(1) + chaoticParticle(t).Position(1) * (VarMax(1) - VarMin(1));
		particle(t).Position(2) = VarMin(2) + chaoticParticle(t).Position(2) * (VarMax(2) - VarMin(2));
		particle(t).Cost = optimizeSVMParkinsonHyperparams(particle(t).Position);
		if particle(t).Cost > GlobalBest.Cost
			GlobalBest.Cost = particle(t).Cost;
            GlobalBest.Position = particle(t).Position;
            bestparticles(nPop/5).Position = GlobalBest.Position;
            bestparticles(nPop/5).Cost = GlobalBest.Cost;
            bestparticles(nPop/5).Best.Position = GlobalBest.Position;
            bestparticles(nPop/5).Best.Cost = GlobalBest.Cost;
			particle(t).Best.Position = particle(t).Position;
            particle(t).Best.Cost = particle(t).Cost;
            break;
		end
	end
    
	VarMin(1) = max(VarMin(1), GlobalBest.Position(1) - rand * (VarMax(1) - VarMin(1)));
	VarMin(2) = max(VarMin(2), GlobalBest.Position(2) - rand * (VarMax(2) - VarMin(2)));

	VarMax(1) = min(VarMax(1), GlobalBest.Position(1) + rand * (VarMax(1) - VarMin(1)));
	VarMax(2) = min(VarMax(2), GlobalBest.Position(2) + rand * (VarMax(2) - VarMin(2)));
	
	%Random expansion strategy
	if abs(VarMax(1) - VarMin(1)) < epsilon2 || VarMin(1) == VarMax(1)
		delta = rand * (VarMax_original(1) - VarMin_original(1)) *0.2;
		VarMin(1) = max(VarMin_original(1), VarMin(1) - delta);
		VarMax(1) = min(VarMax_original(1), VarMax(1) + delta);
	end
	
	%Random expansion strategy
	if abs(VarMax(2) - VarMin(2)) < epsilon2 || VarMin(2) == VarMax(2)
		delta = rand * (VarMax_original(2) - VarMin_original(2)) *0.2; 
		VarMin(2) = max(VarMin_original(2), VarMin(2) - delta);
		VarMax(2) = min(VarMax_original(2), VarMax(2) + delta);
	end
	
    % Create new solutions
    for a = 1:((4 * nPop) / 5)
        particle(a).Position = unifrnd(VarMin, VarMax, VarSize);
        particle(a).Velocity = zeros(VarSize);
        particle(a).Cost = optimizeSVMParkinsonHyperparams(particle(a).Position);
        particle(a).Best.Position = particle(a).Position;
        particle(a).Best.Cost = particle(a).Cost;
        
        if particle(a).Best.Cost > GlobalBest.Cost
            GlobalBest = particle(a).Best;
        end
    end
    
    % Choosing the Five Best Particles
    l = 1;
    for a = ((4 * nPop) / 5) + 1:nPop
        particle(a) = bestparticles(l);
        l = l + 1;
    end
	% Store the Best Cost Value
    BestCosts(it) = GlobalBest.Cost;
	
	xgBest = GlobalBest.Position;  % Global Best Solution from PSO
	xgBestCost = GlobalBest.Cost;

    % Bat Algorithm Section
    
	for i = 1:nPop
		% Adaptive Frequency Update
	    d_i = norm(x(i,:) - xgBest);
		distances2 = arrayfun(@(i) norm(x(i, :) - xgBest), 1:nPop);  % Distance from global best
		d_avg2 = mean(distances2);  % Average distance
		f(i) = Fmin + (Fmax - Fmin) * ((d_i) / ((d_avg2 + epsilon))^beta);

		% Velocity and Position Update
		v(i, :) = v(i, :) + (x(i, :) - xgBest) * f(i);
		s(i, :) = x(i, :) + v(i, :);  % Update Position	

		% Local Search if Random Condition is Met
		if rand > r(i)
			e = -1 + 2 * rand;
			s(i, :) = xgBest + e * mean(A);
		end

		% Apply Boundaries
		s(i, :) = simplebounds(s(i, :), VarMin, VarMax);
		fitnessNew = optimizeSVMParkinsonHyperparams(s(i, :));  % Evaluate new position

		% Accept Solution
		if rand < A(i) && fitnessNew >= fitness(i)
			x(i, :) = s(i, :);  % Update position
			fitness(i) = fitnessNew;
			A(i) = alpha * A(i);  % Update Amplitude
			r(i) = r0 * (1 - exp(-gamma * it));  % Update Pulse Rate
		end

		% Update Global Best Solution in Bat Algorithm
		if fitnessNew >= xgBestCost
			xgBest = s(i, :);
			xgBestCost = fitnessNew;
			GlobalBest.Position = xgBest;
			GlobalBest.Cost = xgBestCost;
		end
	end
		
    % Store Best Cost in Hybrid
    BestCosts(it) = GlobalBest.Cost;
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCosts(it))]);
end

