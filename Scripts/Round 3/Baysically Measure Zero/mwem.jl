


using PrivateMultiplicativeWeights
using CSV
import Plots


weights = CSV.read("our_synthetic_weights.csv");
d = Vector(weights[:weight]);
# Magic number is number of samples (from python notebook)
h = Histogram(d, 38462)

epsilons = reverse([10.0^-n for n in range(1, stop=6)])
mses = []
max_errors = []

for ε in epsilons
    println("Running MWEM with ε=$ε")
    mw = mwem(SeriesRangeQueries(length(d)), h, MWParameters(epsilon=ε, iterations=500));
    mse = mean_squared_error(mw)
    println("Mean squared error = ", mse)
    append!(mses, mse)
end

Plots.plot(epsilons, mses, xlabel = "Epsilon", ylabel="MSE", title="Utility vs Privacy", xaxis=:log)
