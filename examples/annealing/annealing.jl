# # Default parameters for Kernel Correlation
# While the function [`rigid_kc`](@ref) has a couple of parameters to set,
# users can mostly rely on the defaults.
# These we want to justify here.
#
# To be more precise:
# How should we choose the parameters such that it is most likely to find the
# global optimum in single restart, and then how many restarts should we
# perform?
# The parameters to be determined are:
# * `scale`: the annealing plan for the scale parameter ``\sigma``,
# * `smm`: if and how to perform stochastic majorization minimization, and
# * `iterations`: how many majorization minimization iterations to run.
# For `restarts` we will then choose `RandomRestarts(n)` with an `n` as small
# as possible.
# The remaining arguments (`axis_align`, `report_iteration`, and
# `report_restart`) are merely for performance/interactivity reasons.

# ## Packages

using Revise #src
using PointCloudRegistration
using Rotations
using GLMakie
using Format
using Combinatorics
using TypedTables
using Random
using LinearAlgebra
using Statistics

# ## Setup
# Usually, we are only interested in the best result `rigid_kc` finds over
# all restarts.
# Here, however, it makes a difference if it found a good result only once (we
# got lucky!) or basically all the time (the choice of parameters almost
# guarantees finding this solution).
# To analyse the results of the individual restarts, we can hook into the
# algorithm via the `report_restart` keyword argument which accepts a specific
# callback.
# Ours will have to be able to collect the results of all the restarts and
# also, for convenience, store the lowest cost this run has achieved.

mutable struct RestartCollector{T}
    transformations::Vector{T}
    mincost
end
RestartCollector() = RestartCollector([], Inf)

function (rc::RestartCollector)(; transformation, cost, kwargs...)
    push!(rc.transformations, transformation)
    rc.mincost = min(rc.mincost, cost)
end

# We can now use objects of type `RestartCollector` as callbacks for the
# `report_restart` argument.
# With this, we define the core evaluation function.
# It takes the source and target for the registration, the number of MM
# iterations and how many samples to take in the stochastic MM (where a
# non-positive number means to do no stochastic MM).
# Since we provide the target in the "prepared" version, this already includes
# info about `scale`.
# To ensure that our analysis is not affected by randomness, we fix the two
# sources of randomness, the sampling for SMM and the restart initializations,
# by using random number generators with set seeds.

function eval_params(; source, prepd_target, iterations, smm_count)
    restart_collector = RestartCollector()
    seed = 2
    smm = if smm_count > 0
        Smm(smm_count, Xoshiro(-seed))
    else
        NoSmm()
    end
    T = rigid_kc(
        source,
        prepd_target;
        restarts = RandomRestarts(1000, Xoshiro(seed)),
        iterations,
        smm,
        report_restart = restart_collector,
    )
    return (;
        T,
        cost = restart_collector.mincost,
        restartTs = restart_collector.transformations,
    )
end

# The function runs the registration and afterwards gives back the best found
# transformation `T`, its assosicated `cost`, and all transformations that were
# found in the individual restarts (`restartTs`).

# ## Search space
# Next, let us define what choices for the three parameters we want to consider.
# ### Iterations
# Here, we simply try to cover a sensible range with a few numbers:
searchspace_iterations = [10, 50, 100, 200, 500];
# searchspace_iterations = [10, 50];

# ### Stochastic MM
# In principle, six points are enough to uniquely define a rigid transformation
# in 3D.
# So this is a lower bound for how many samples we need for stochastic MM.
# On the other hand, one hundred points seem to be overly sufficient.
# Let us therefore set
searchspace_smm_count = [0, 10, 50, 100];
# searchspace_smm_count = [0, 10];
# (remember that `0` means performing no SMM).

# ### Annealing scales
# This is the trickiest one since it is typically more than one number and it
# also depends on the scale of the point clouds.
# In general, we make the following assumptions:
# First, the scale should decrease during annealing.
# Second, if we want to optimize the kernel correlation for a specific scale,
# we should perform the final optimization with that scale.
#
# Let us thus employ the following strategy:
# For all potential scales supposed to use in the annealing, put the smallest
# one aside and produce every subset of the remaining ones.
# Sort each of these sets descendingly and append the smallest one.

function searchspace_scale(potential_scales)
    potential_scales = sort(potential_scales; rev = true)
    min_scale = potential_scales[end]
    other_scales = potential_scales[begin:end - 1]
    return [[subset; min_scale] for subset in powerset(other_scales)]
end

# ## The search
# We can put together what we set up so far and define a function that takes
# two point clouds and analyzes the different ways to rigidly register them
# via kernel correlation.
#
# It first builds the search space for the annealing and then enumerates the
# whole search space using nested iteration.
# Note that we prepare the target for kernel correlation registration only once
# per choice of `scale` to save time.
# We also run everything in parallel that depends on different scales.
#
# After we have collected all the information from all runs into one long table,
# we find the overall best solution (the one with the lowest cost).
# Assuming that *any* of the tested strategies is the "right" one, this solution
# should be the global optimum.
#
# Having this global optimum, we can then determine for every restart if it was
# a success.
# If the transformation found in a particular restart is ``T`` and the global
# optimum is some transformation ``U``, then ``T \circ U^{-1}`` should be
# (close to) the identity transformation if ``T`` is also (near) the global
# optimum.
# We hence consider a restart successful if the rotation angle of
# ``T \circ U^{-1}`` is below 5° and its translation norm is below the minimum
# scale used for annealing.
#
# We compute the proportion or restarts that are successful for every
# combination of registration parameters and call this its success rate.
#
# However, it is not immediately clear how to interpret this success rate.
# Should we strive for a near 100 % rate or is 10 % fine because we can afford
# some more restarts?
# Since we are interested in the number of restarts as stated in the beginning,
# let us convert the success rate into that number.
# Say, we have a success rate ``p`` per restart and we want a probability ``q``
# that we achieved success after ``k`` restarts.
# This means:
# ```math
# 1 - (1 - p)^k &> q \\
# (1 - p)^k &< 1 - q \\
# k \log(1 - p) &< log(1 - q) \\
# k &> log(1 - q) / log(1 - p)
# ```
# Setting ``q`` to 99 % allows us to perform the wanted conversion.
#
# In the end, the function returns the complete data set as well as the best
# found transformation.

function analyze(source, target)
    min_scale = PointCloudRegistration.avg_nn_dist(target)
    max_scale = sqrt(PointCloudRegistration.maxcoveigval(target))
    potential_scales = range(min_scale, max_scale; length = 7)
    tasks = Task[]
    for scale in searchspace_scale(potential_scales)
        task = Threads.@spawn begin
            table = Table(smm_count = [], scale = [], iterations = [], T = [], cost = [], restartTs = [])
            prepd_target = prepare_target_kc(target; scale)
            for smm_count in searchspace_smm_count
                for iterations in searchspace_iterations
                    result = eval_params(; source, prepd_target, iterations, smm_count)
                    push!(table, (; smm_count, scale, iterations, result...))
                end
            end
            table
        end
        push!(tasks, task)
    end
    data = FlexTable(mapreduce(fetch, vcat, tasks))
    bestT = argmin(r -> r.cost, data).T
    invbestT = inv(bestT)
    data.success_rate = map(data) do row
        success_count = count(row.restartTs) do T
            diffT = T ∘ invbestT
            rot_angle = diffT.linear |> rotation_angle |> rad2deg
            trl_norm = norm(diffT.translation)
            rot_angle < 5 && trl_norm < min_scale
        end
        success_count / length(row.restartTs)
    end
    neginf2inf(x) = isinf(x) && x < zero(x) ? typemax(x) : x
    data.necessary_restarts = neginf2inf.(log(.01) ./ log.(1 .- data.success_rate))
    return data, bestT
end

# # Evaluation on protein structures
# Let us now run the `analyze` function on actual data.
# As a first candidate, we can use two adenylate kinase structures (PDB IDs
# `1ake` and `4ake`):

X, Y = PointCloudRegistration.Assets.load_1ake_A_4ake_A()

# This is how they look:

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    plot!(ax, X)
    plot!(ax, Y)
    animate_plot_rotation(fig, ax)
end

# And now we do the analysis:

data, bestT = analyze(Y, X);

# To check that any of the tested strategies produced a good result, we can
# plot the target together with the registered source:

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    plot!(ax, X)
    plot!(ax, bestT(Y))
    fig
end

# Having performance in mind, we are interested in strategies that need at most
# one hundred restarts and two different scales for the annealing.

promising = filter(data) do row
    row.necessary_restarts < 100 && length(row.scale) <= 2
end;

include("plot.jl")


