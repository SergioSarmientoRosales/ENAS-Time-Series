%Code created by (1) Martha Isabel Escalona Llaguno & (2) Sergio M. Sarmiento-Rosales.
           %Universidad Autónoma de Zacatecas / Tec. de Monterrey
                   %For AutoML 2024. Free use license
            %Contacts: (1) ing_miell@hotmail.com (2) sarmiento@tec.mx

% CLEAR WORKSPACE
clear all;
close all;
clc;
% Start the timer
tic;
% RANDOM SEED
% The random seed  is used to initialize the random number generator (RNG) to ensure reproducibility of the results.
rng(1);

% LOAD DATASET
% These datasets contain only 2 columns, the first for the dates and the second for the variable. We will only use the variable.
% You can use any time series dataset

% Prompt the user to choose between random and uniform population
choice_population = menu('Select Dataset:', 'PJM_Load_hourly', 'PJME_hourly', 'PJMW_hourly', 'NI_hourly', 'FE_hourly', 'EKPC_hourly', 'DUQ_hourly', 'DOM_hourly', 'DEOK_hourly', 'DAYTON_hourly', 'COMED_hourly', 'AEP_hourly', 'Custom Dataset');

switch choice_population
    case 1 
         filename = 'PJM_Load_hourly.csv';       
    case 2 
         filename = 'PJME_hourly.csv';
    case 3
         filename = 'PJMW_hourly.csv';
    case 4
         filename = 'NI_hourly.csv';
    case 5
         filename = 'FE_hourly.csv';
    case 6
         filename = 'EKPC_hourly.csv';
    case 7
         filename = 'DUQ_hourly.csv';
    case 8
         filename = 'DOM_hourly.csv';
    case 9
         filename = 'DEOK_hourly.csv';
    case 10
         filename = 'DAYTON_hourly.csv';
    case 11
         filename = 'COMED_hourly.csv';      
    case 12
         filename = 'AEP_hourly.csv';
    case 13
         filename = input('Enter your Dataset (example: PJMW_hourly.csv) =', 's'); % 's' indicates input as string
    otherwise
         error('Invalid choice. Please select a valid dataset.');
end

Set = readtable(filename);

% Data without date
%only for datasets with date, if you dataset has not date, you can delete
%or coment this part
secondColumn = table2cell(Set(:, 2));
[DataSet, settings] = mapminmax(cell2mat(secondColumn'));
Normalized_DataSet = num2cell(DataSet);
%%%%%


% INITIAL CONFIGURATION
num_features = size(Normalized_DataSet', 2); % NUMBER OF FEATURES

% Ask the user to choose between custom or default values
choice = input('Choose:\n1. Enter custom values\n2. Use default values\nEnter your choice: ');

while true
    if choice == 1
        % Request user input for the initial population
        num_individuals = input('Enter the number of individuals in the initial population: ');

        % Request user input for the number of generations
        num_generations = input('Enter the number of generations: ');

        % Request user input for the elitism rate (value between 0 and 1)
        elitism_rate = input('Enter the elitism rate (value between 0 and 1): ');
        while elitism_rate < 0 || elitism_rate > 1
            elitism_rate = input('Invalid input! Enter the elitism rate (value between 0 and 1): ');
        end

        % Request user input for the crossover probability (value between 0 and 1)
        crossover_probability = input('Enter the crossover probability (value between 0 and 1): ');
        while crossover_probability < 0 || crossover_probability > 1
            crossover_probability = input('Invalid input! Enter the crossover probability (value between 0 and 1): ');
        end

        % Request user input for the mutation probability (value between 0 and 1)
        mutation_probability = input('Enter the mutation probability (value between 0 and 1): ');
        while mutation_probability < 0 || mutation_probability > 1
            mutation_probability = input('Invalid input! Enter the mutation probability (value between 0 and 1): ');
        end
        
        break; % Exit the loop if all inputs are valid

    elseif choice == 2
        % Set default values
        num_individuals = 20; % INITIAL POPULATION
        num_generations = 5; % NUMBER OF GENERATIONS
        elitism_rate = 0.2; % RATE OF ELITISM
        crossover_probability = 0.9; % 90%
        mutation_probability = 0.1; % 10%
        break; % Exit the loop if default values are used
    else
        % Display error message for invalid input
        choice = input('Error: Please enter either 1 or 2 to choose custom or default values. Enter your choice: ');
    end
end

% Display the values entered by the user or default values
fprintf('Configured parameters:\n');
fprintf('Number of individuals: %d\n', num_individuals);
fprintf('Number of generations: %d\n', num_generations);
fprintf('Elitism rate: %.2f\n', elitism_rate);
fprintf('Crossover probability: %.2f\n', crossover_probability);
fprintf('Mutation probability: %.2f\n', mutation_probability);


% BOUNDS OF SEARCH SPACE
max_lags = 20;
min_lags = 1;
max_neurons = 20;
min_neurons = 1;
%%%%%%%%%%%%%



%INITIALIZATION
population = cell(num_individuals, 1);
population_decoded = cell(num_individuals, 1);

% Prompt the user to choose between random and uniform population
choice_population = menu('Select population type:', 'Random Population', 'Uniform Population');

% Check the user's choice and generate the population accordingly
switch choice_population
    case 1 % Random Population
        % RANDOM POPULATION
        for i = 1:num_individuals
            % GENERATE RANDOM GENOTYPE
            neurons = randi([min_neurons, max_neurons], 1, num_features); % DEFINE THE NUMBER OF NEURONS IN THE HIDDEN LAYER
            lags = randi([min_lags, max_lags], 1, num_features); % DEFINE THE NUMBER OF LAGS

            % ENCODE THE GENOTYPE USING GRAY CODE
            neurons_gray = rstr_to_bstr(neurons);
            lags_gray = rstr_to_bstr(lags);
            genotype_gray = [neurons_gray, lags_gray];

            % SAVE THE INDIVIDUAL IN THE POPULATION
            population{i} = genotype_gray;
            % SAVE THE INDIVIDUAL IN THE POPULATION WITHOUT ENCODE
            population_decoded{i} = [neurons, lags];
                        
        end
            figure;
            pop =cell2mat(population_decoded(:, 1));
            neurons_plot = (pop(:,1));
            lags_plot = (pop(:,end));
            scatter(neurons_plot,lags_plot)
            grid on
            xlabel('Hidden Neurons')
            ylabel('Lags')
            title('Random Population')
            xlim([1, 20]);

           

    case 2 % Uniform Population
        % UNIFORM POPULATION
        for i = 1:num_individuals
            % GENERATE GENOTYPE
            neurons = i; % DEFINE THE NUMBER OF NEURONS IN THE HIDDEN LAYER
            lags = i; % DEFINE THE NUMBER OF LAGS

            % ENCODE THE GENOTYPE USING GRAY CODE
            neurons_gray = rstr_to_bstr(neurons);
            lags_gray = rstr_to_bstr(lags);
            genotype_gray = [neurons_gray, lags_gray];

            % SAVE THE INDIVIDUAL IN THE POPULATION
            population{i} = genotype_gray;
            % SAVE THE INDIVIDUAL IN THE POPULATION WITHOUT ENCODE
            population_decoded{i} = [neurons, lags];
        end

         figure;
            pop =cell2mat(population_decoded(:, 1));
            neurons_plot = (pop(:,1));
            lags_plot = (pop(:,end));
            scatter(neurons_plot,lags_plot)
            grid on
            xlabel('Hidden Neurons')
            ylabel('Lags')
            title('Uniform Population')
            xlim([1, num_individuals]);
end




% STRUCTURE TO STORE RESULTS OF EACH GENERATION
generation_results = cell(num_generations, 1);


% MAIN LOOP FOR THE EVOLUTIONARY ALGORITHM
for gen = 1:num_generations
    clc
    disp(['Generation: ', num2str(gen)]); % SHOWS THE ACTUAL GENERATION
    training_results = cell(num_individuals, 3); % Stores R_test, neurons_decoded, lags_decoded

    for i = 1:num_individuals
        
        % DECODE THE POPULATION
        [neurons_decoded, lags_decoded] = bstr_to_rstr_inv(population{i});
        
        % DUE TO THE NATURE OF EVOLUTIONARY ALGORITHMS, WE ADD A LIMIT TO PREVENT A MUTATION FROM RESULTING IN 0 
        if neurons_decoded < 1
               neurons_decoded = 1;
        end
        if lags_decoded < 1
               lags_decoded = 1;
        end
              
        % CREATE THE MODEL
        net = narnet(1:lags_decoded, neurons_decoded);

        % Split data into training, validation, and test sets
        net.divideFcn = 'divideblock'; % Divide by blocks
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;

        % PREPARE DATA FOR TRAINING
        [Xs, Xi, Ai, Ts] = preparets(net, {}, {}, Normalized_DataSet); % TIME SERIES DATASET

        % TRAIN THE NET
              
        [net, tr] = train(net, Xs, Ts, Xi, Ai);

        % Make predictions on test data
        testInd = tr.testInd;
        testInputs = Xs(:, testInd);
        testTargets = Ts(:, testInd);

        % Simulate the network for test data
        testOutputs = net(testInputs, Xi, Ai);

        % Convert cells to matrices for regression calculation
        testTargetsMatrix = cell2mat(testTargets);
        testOutputsMatrix = cell2mat(testOutputs);

        % Calculate the correlation coefficient R between actual outputs and predictions
        R_test = corrcoef(testTargetsMatrix, testOutputsMatrix);
        R_test = R_test(1, 2); % Get the correlation coefficient

        % Calculate mean squared error (MSE) on test data
        MSE_test = mse(testTargetsMatrix, testOutputsMatrix);

        training_results{i, 1} = R_test; % Correlation coefficient R
        training_results{i, 2} = neurons_decoded;
        training_results{i, 3} = lags_decoded;
        training_results{i, 4} = MSE_test;
        disp(['Individual ', num2str(i) , ' Trained.   ', 'R = ' , num2str(R_test),  '     Hidden Neurons = ', num2str(neurons_decoded) , '    Lags = ',num2str(lags_decoded)]);
    end
    

    % SAVE RESULTS
    generation_results{gen} = training_results;
    mean_train(gen) = mean(cell2mat(generation_results{gen}(:, 1)));
    
    % Obtain evaluation metrics for each individual in the population
    evaluation_metrics = cellfun(@(x) x(1), training_results);

    % Sort the population indices by evaluation metrics
    [~, sorted_indices] = sort(evaluation_metrics, 'descend');

    % Select the best parents (top individuals based on elitism rate)
    num_elites = round(elitism_rate * num_individuals);
    best_parents_indices = sorted_indices(1:num_elites);
    best = sorted_indices(1);
    best_result(gen) = training_results{best, 1};
    best_lag(gen) = training_results{best, 3};
    best_neuron(gen) = training_results{best, 2};

    disp(['Best evaluation metric (R): ', num2str(evaluation_metrics(best))]);
    disp(['Best lags: ', num2str(best_lag(gen))]);
    disp(['Best neurons: ', num2str(best_neuron(gen))]);
    pause(3)

    clc
    
   
    % Introduce random individuals to maintain diversity
    for i = num_individuals-num_elites+1:num_individuals
        neurons = randi([min_neurons, max_neurons], 1, num_features);
        lags = randi([min_lags, max_lags], 1, num_features);
        neurons_gray = rstr_to_bstr(neurons);
        lags_gray = rstr_to_bstr(lags);
        population{i} = [neurons_gray, lags_gray];
    end

    % Crossover and Mutation
    for k = 1:2:num_individuals-1
        % Select two parents using tournament selection
        parent1 = population{best_parents_indices(randi(num_elites))};
        parent2 = population{best_parents_indices(randi(num_elites))};

        % probability for crossover
        if rand < crossover_probability
            % Choose two random crossover points
            n = numel(parent1);
            crossover_point1 = randi(n);
            crossover_point2 = randi(n);

            % Ensure the second crossover point is greater than the first
            if crossover_point1 > crossover_point2
                temp = crossover_point1;
                crossover_point1 = crossover_point2;
                crossover_point2 = temp;
            end

            % Perform crossover at the crossover points
            child1 = [parent1(1:crossover_point1), parent2(crossover_point1+1:crossover_point2), parent1(crossover_point2+1:end)];
            child2 = [parent2(1:crossover_point1), parent1(crossover_point1+1:crossover_point2), parent2(crossover_point2+1:end)];
            
            % Save the children in the population
            population{sorted_indices(k+1)} = child1;
            population{sorted_indices(k+2)} = child2;
        end
         % probability for mutation
        if rand < mutation_probability
            % Mutation: BitFlipMutation with 10% probability
            child1 = BitFlipMutation(parent1, mutation_probability);
            child2 = BitFlipMutation(parent2, mutation_probability);

            % Save the mutated children in the population
            population{sorted_indices(k+1)} = child1;
            population{sorted_indices(k+2)} = child2;
        end
    end

end

% Initialize a variable to store the global minimum of the first column
All_Results = vertcat(generation_results{:});
% Initialize the global minimum value
global_min_value = Inf; % Initialize to infinity or a large value);

% Iterate over each generation
for gen = 1:num_generations
    % Get the data for the current generation
    current_gen_data = generation_results{gen, 1};
    
    % Iterate over each row of the first column and find the minimum
    for i = 1:num_individuals
        % Convert the cell to a matrix and find the minimum in the first column
        cell_min = min(cell2mat(current_gen_data(i, 1)));
        if cell_min < global_min_value
            global_min_value = cell_min;
        end
    end
end

% Vector to store the averages for each generation
average_generations = zeros(num_generations, 1);

% Calculate the average for each generation
for i = 1:num_generations
    average_generations(i) = mean(cell2mat(generation_results{i, 1}(:, 1)));
end

% Create a new figure for the histograms
figure;

% Histogram for the first dataset
subplot(2, 2, 1);
histogram(cell2mat(generation_results{1, 1}(:,2)), 'BinLimits', [1, 20], 'BinMethod', 'integers');
xlabel('Number of Hidden Neurons');
ylabel('Frequency');
xlim([1, 20]); % Set upper limit for x-axis
ylim([0, 20]); % Set upper limit for y-axis
grid on
title('Frequency of Hidden Neurons in First Generation');

% Histogram for the second dataset
subplot(2, 2, 2);
histogram(cell2mat(generation_results{1, 1}(:,3)), 'BinLimits', [1, 20], 'BinMethod', 'integers');
xlabel('Number of Lags');
ylabel('Frequency');
xlim([1, 20]); % Set upper limit for x-axis
ylim([0, 20]); % Set upper limit for y-axis
grid on
title('Frequency of Lags in First Generation');

% Histogram for the third dataset
subplot(2, 2, 3);
histogram(cell2mat(generation_results{num_generations, 1}(:,2)), 'BinLimits', [1, 25], 'BinMethod', 'integers');
xlabel('Number of Hidden Neurons');
ylabel('Frequency');
xlim([1, max(cell2mat(generation_results{num_generations, 1}(:,2)))+1]); % Set upper limit for x-axis
ylim([0, 20]); % Set upper limit for y-axis
grid on
title('Frequency of Hidden Neurons in Final Generation');

% Histogram for the fourth dataset
subplot(2, 2, 4);
histogram(cell2mat(generation_results{num_generations, 1}(:,3)), 'BinLimits', [1, 25], 'BinMethod', 'integers');
xlabel('Number of Lags');
ylabel('Frequency');
xlim([1, max(cell2mat(generation_results{num_generations, 1}(:,3)))+1]); % Set upper limit for x-axis
ylim([0, 20]); % Set upper limit for y-axis
grid on
title('Frequency of Lags in Final Generation');

figure
scatter(cell2mat(generation_results{num_generations, 1}(:,2)),cell2mat(generation_results{num_generations, 1}(:,3)))
grid on
xlabel('Hidden Neurons')
ylabel('Lags')
title('Final Population')
xlim([1, max(cell2mat(generation_results{num_generations, 1}(:,2)))+1]);

%%%%%%%%%%HERE IS THE PLOT CODE, YOU CAN CHOOSE THE PLOT
BEST_NEURAL_NET =[best_result;best_neuron;best_lag];
BEST_NEURAL_NET =BEST_NEURAL_NET';
% Find the index of the maximum value in column 1
[~, idx_max] = max(BEST_NEURAL_NET(:, 1));

% Get the corresponding values in columns 2 and 3
max_value_col1 = BEST_NEURAL_NET(idx_max, 1);
N_hidden= BEST_NEURAL_NET(idx_max, 2);
Lags = BEST_NEURAL_NET(idx_max, 3);

% Print the results

disp(['The Best Individual is at Generation: ', num2str(idx_max) , '     R = ' , num2str(max_value_col1),  '   Hidden Neurons = ', num2str(N_hidden) , '    Lags = ',num2str(Lags)]);






% Initialize the choice variable outside the loop
choice = 0;

% Loop until the user chooses to exit
while choice ~= 4
        % Prompt the user to choose an option
        choice = menu('Select an option:', 'Plot R in All Generations', 'Plot R in First and Final Generations', 'R Best Individual in Generation', 'Do Not Plot');
        
        % Check the user's choice and perform the corresponding action
        switch choice
            case 1 % Plot All Generations
                 %%%%%%%%%%%%%%%%%%%
                
                % Initialize variables to store max, min, and average values
                max_values = zeros(num_generations, 1);
                min_values = zeros(num_generations, 1);
                avg_values = zeros(num_generations, 1);
                
                % Iterate over each generation
                for gen = 1:num_generations
                    % Extract the data for the current generation
                    current_data = cell2mat(generation_results{gen, 1}(:,1));
                    
                    % Calculate maximum, minimum, and average values for the current generation
                    max_values(gen) = max(current_data);
                    min_values(gen) = min(current_data);
                    avg_values(gen) = mean(current_data);
                end
                
                %Uncomment this part if you want to graph the results of all generations.
                % Calculate the overall minimum value across all generations
                min_y_limit = min(min_values);
                
                % Create a new figure for the plots
                figure;
                
                % Iterate over each generation to plot the data
                for gen = 1:num_generations
                    % Extract the data for the current generation
                    current_data = cell2mat(generation_results{gen, 1}(:, 1));
                    
                    % Plot the current generation data
                    subplot(num_generations, 1, gen);
                    plot(current_data);
                    hold on;
                    plot([1, length(current_data)], [avg_values(gen), avg_values(gen)], 'g--', 'LineWidth', 1); % Line for average
                    hold off;
                    xlabel('Individual');
                    ylabel('R');
                    title(['R in Generation ', num2str(gen)]);
                    legend('Data', 'Average');
                    xlim([1, num_individuals]);
                    ylim([min_y_limit, 1]); % Set y-axis limits for each subplot
                    grid on;
                end
                
                % Adjust spacing between subplots
                sgtitle('R in Each Generation');
            case 2 % Plot First and Final Generations
                % Define the generations to plot: first and final
                    generations_to_plot = [1, num_generations];
                    
                    % Create a new figure for the plots
                    figure;
                    
                    % Define the generations to plot: first and final
                    generations_to_plot = [1, num_generations];
                    
                    % Iterate over the specified generations to plot the data
                    for idx = 1:length(generations_to_plot)
                        gen = generations_to_plot(idx);
                        
                        % Extract the data for the current generation
                        current_data = cell2mat(generation_results{gen, 1}(:, 1));
                        
                        % Plot the current generation data
                        subplot(2, 1, idx);
                        plot(current_data);
                        hold on;
                        
                        % Plot the average line
                        plot([1, length(current_data)], [avg_values(gen), avg_values(gen)], 'g--', 'LineWidth', 1);
                        
                        % Find and plot the maximum value point
                        [max_value, max_index] = max(current_data);
                        plot(max_index, max_value, 'ro', 'MarkerSize', 8, 'LineWidth', 2); % Red point at the maximum value
                        
                        % Add a text label for the maximum value
                        text(max_index, max_value, ['Max: ', num2str(max_value)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
                        
                        hold off;
                        xlabel('Individual');
                        ylabel('R');
                        title(['R in Final Generation ']);
                        legend('Data', 'Average', 'Max Value');
                        xlim([1, num_individuals]);
                        ylim([min_y_limit, 1]); % Set y-axis limits for each subplot
                        grid on;
                    end
                    
                    % Adjust spacing between subplots
                    sgtitle('R in First and Final Generation');
            case 3 % Plot Something Else
                % Create the initial plot 
                %BEST INDIVIDUAL IN EACH GENERATION
                figure;
                plot(best_result);
                xlabel('Best individual in Generation');
                ylabel('R');
                title('Best Individuals Over Generations');
                grid on
                % Find the index of the maximum point in best_result
                [max_value, max_index] = max(best_result);
                
                % Highlight the maximum point on the plot
                hold on; % Keep the current plot
                plot(max_index, max_value, 'ro', 'MarkerSize', 10, 'LineWidth', 2); % Red point at the maximum
                text(max_index, max_value, ['Max Value: ', num2str(max_value)], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left'); % Label for the maximum
                hold off; % Release the "hold on" mode
                       
            case 4 % Do Not Plot
                % Stop the timer and get the elapsed time
                elapsed_time = toc;
                disp(['Elapsed time: ', num2str(elapsed_time), ' seconds']);
                return; % Exit the script without plotting anything
        end
end





function mutated_individual = BitFlipMutation(individual, probability)
    % Perform Bit Flip Mutation on the individual with a certain probability
    mutated_individual = individual;
    for k = 1:numel(individual)
        if rand < probability
            % Invert the bit at position k
            mutated_individual(k) = ~mutated_individual(k);
        end
    end
end

function gray_code = rstr_to_bstr(neurons)
    % Verify that the number of neurons is within the allowed range
    if neurons < 0 || neurons > 20
        error('The number of neurons must be in the range of 0 to 20.');
    end
    
    % Convert to 5-bit binary representation (to cover the range 0-20)
    binary = dec2bin(neurons, 5) - '0';

    % Calculate the Gray code
    gray_code = [binary(1), xor(binary(1), binary(2)), xor(binary(2), binary(3)), xor(binary(3), binary(4)), xor(binary(4), binary(5))];
end

function [neurons_decoded, lags_decoded] = bstr_to_rstr_inv(genotype_gray)
    % Verify the length of the genotype
    if length(genotype_gray) ~= 10
        error('The length of the genotype must be 10 (5 bits for neurons and 5 bits for lags).');
    end
    
    % Decode neurons from the Gray code
    neurons_binary = zeros(1, 5);
    neurons_binary(1) = genotype_gray(1);
    for i = 2:5
        neurons_binary(i) = xor(neurons_binary(i-1), genotype_gray(i));
    end
    neurons_decoded = bin2dec(num2str(neurons_binary));

    % Decode lags from the Gray code
    lags_binary = zeros(1, 5);
    lags_binary(1) = genotype_gray(6);
    for i = 2:5
        lags_binary(i) = xor(lags_binary(i-1), genotype_gray(i+5));
    end
    lags_decoded = bin2dec(num2str(lags_binary));
end










