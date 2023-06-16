import yfinance as yf
import talib
import random

# Define the parameters for your genetic algorithm
population_size = 50
generations = 100
mutation_rate = 0.1

# Define the available technical indicators
technical_indicators = ["SMA", "EMA", "RSI", "MACD", "BBANDS", "DMI+", "DMI-", "ADX"]

# Define the chromosome representation
chromosome_length = len(technical_indicators)

# Define the trading symbol and date range
symbol = "BIKAJI.NS"
ticker = yf.Ticker(symbol)
stock_data = ticker.history(period="15d",interval="15m")



def calculate_fitness(chromosome):
    # Calculate the technical indicators based on the selected chromosome
    indicator_data = {}
    for indicator_group in chromosome:
        for indicator in indicator_group:
            if indicator == "SMA":
                sma = talib.SMA(stock_data["Close"], timeperiod=20)
                indicator_data["SMA"] = sma
            elif indicator == "EMA":
                ema = talib.EMA(stock_data["Close"], timeperiod=20)
                indicator_data["EMA"] = ema
            elif indicator == "RSI":
                rsi = talib.RSI(stock_data["Close"], timeperiod=14)
                indicator_data["RSI"] = rsi
            elif indicator == "MACD":
                macd, _, _ = talib.MACD(stock_data["Close"])
                indicator_data["MACD"] = macd
            elif indicator == "BBANDS":
                upper, middle, lower = talib.BBANDS(stock_data["Close"])
                indicator_data["BBANDS_upper"] = upper
                indicator_data["BBANDS_middle"] = middle
                indicator_data["BBANDS_lower"] = lower

    # Calculate the DMI+ and DMI- from high and low prices
    high = stock_data["High"]
    low = stock_data["Low"]
    dmi_plus = talib.PLUS_DM(high, low)
    dmi_minus = talib.MINUS_DM(high, low)
    indicator_data["DMI+"] = dmi_plus
    indicator_data["DMI-"] = dmi_minus

    # Calculate the ADX based on DMI+ and DMI-
    adx = talib.ADX(indicator_data["DMI+"], indicator_data["DMI-"], stock_data["Close"])
    indicator_data["ADX"] = adx

    # Extract the DMI+ and DMI- from indicator data
    dmi_plus = indicator_data["DMI+"]
    dmi_minus = indicator_data["DMI-"]

    # Calculate the ADX from indicator data
    adx = indicator_data["ADX"]

    # Implement your own criteria to determine the fitness score based on the indicators
    # Example: Calculate the cumulative returns
    cumulative_returns = stock_data["Close"].pct_change().cumsum()

    # Apply your strategy rules and calculate fitness score
    # Example: Buy when DMI+ crosses above DMI- and ADX is above a certain threshold
    buy_signals = (dmi_plus > dmi_minus) & (adx > 25)
    sell_signals = (dmi_plus < dmi_minus) & (adx > 25)

    # Calculate the fitness score based on the strategy's performance
    fitness_score = cumulative_returns[-1] * sum(buy_signals) / sum(sell_signals)

    return fitness_score


def generate_chromosome():
    chromosome = []

    for _ in range(chromosome_length):
        indicator_group = random.choice(technical_indicators)
        if isinstance(indicator_group, list):
            chromosome.extend(indicator_group)
        else:
            chromosome.append(indicator_group)

    return chromosome


# Generate an initial population
def generate_population():
    population = []
    for _ in range(population_size):
        chromosome = generate_chromosome()
        population.append(chromosome)
    return population


# Perform crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, chromosome_length - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Perform mutation
def mutation(chromosome):
    for i in range(chromosome_length):
        if random.random() < mutation_rate:
            chromosome[i] = random.choice(technical_indicators)
    return chromosome

# Run the genetic algorithm
population = generate_population()
for generation in range(generations):
    #print(f"Generation {generation+1}")

    # Evaluate the fitness of each chromosome in the population
    fitness_scores = []
    for chromosome in population:
        fitness_scores.append(calculate_fitness(chromosome))

    # Select chromosomes for reproduction (based on fitness scores)
    selected_population = []
    for _ in range(population_size):
        selected_chromosome = random.choices(population, weights=fitness_scores)[0]
        selected_population.append(selected_chromosome)

    # Create the next generation through crossover and mutation
    offspring_population = []
    for _ in range(population_size // 2):
        parent1 = random.choice(selected_population)
        parent2 = random.choice(selected_population)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutation(child1)
        child2 = mutation(child2)
        offspring_population.append(child1)
        offspring_population.append(child2)

    # Replace old population with the new population
    population = offspring_population

# Extract the best-performing strategy
best_chromosome = population[0]

best_fitness = calculate_fitness(best_chromosome)

# Perform any additional steps like backtesting, validation, and fine-tuning

# Print the best strategy and its fitness score
print(f"Best Strategy: {symbol} {best_chromosome}")
print("Fitness Score:", best_fitness)
