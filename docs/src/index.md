# PointCloudRegistration.jl

**PointCloudRegistration.jl** is a package for performing rigid and nonrigid
registration of point clouds, also known as *superimposing* or *aligning*.

## Introductory example

To understand what point cloud registration is and why we need it, let's have
a look at these two cute cats:

```@setup 1
using PointCloudRegistration
using GLMakie
```

```@repl 1
using PointCloudRegistration
source, target = PointCloudRegistration.Assets.load_cats();
```

```@setup 1
let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    hidedecorations!(ax)
    plot!(ax, source; label = "source")
    plot!(ax, target; label = "target")
    axislegend(ax)
    save("source-target.png", fig)
end
```

![](source-target.png)

They are both represented as weighted (indicated by marker size) point clouds.
As it is common in the context of registration, we call one of them the *source*
and the other the *target*.
While the cats actually only differ in the position of their tails, the two
point clouds are considerably more different.
If you were to compare them point coordinate by point coordinate, you would
pick up lots of differences that are not related to the tail position.
Point cloud registration aims at resolving this issue.

We can inspect the numerical representation of the cats:

```@repl 1
source
target
```

To perform registration, we can first find a *rigid transformation* that rotates
and translates the source such that both point clouds sit on top of each other:

```@repl 1
rigid_transformation = rigid_registration(source, target)
```

We can apply `rigid_transformation` to `source` and obtain a new point cloud:

```@repl 1
source2 = rigid_transformation(source)
```

Plotting both `source2` and `target` shows that the two are aligned well:

```@setup 1
let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    hidedecorations!(ax)
    plot!(ax, source2; label = "rigidly registered source")
    plot!(ax, target; label = "target")
    axislegend(ax)
    save("rr-source-target.png", fig)
end
```

![](rr-source-target.png)

Often, we are already done here.
However, we might be interested in explaining more of the differences between
the point clouds, for which we turn to _non-rigid registration_:

```@repl 1
nonrigid_displacement =  nonrigid_registration(
  source2,
  target,
  DistancePreserving(; max_edge_length = 45, sensitivity = 1.4, rel_deviation = 1e-4),
)
```

As we can see, this requires a bit more domain knowledge and fine-tuning of
parameters compared to rigid registration.
Applying the result to `source2` provides a new point cloud:

```@repl 1
source3 = nonrigid_displacement(source2)
```

The plot shows the final result:

```@setup 1
let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    hidedecorations!(ax)
    plot!(ax, source3; label = "rigidly and non-rigidly registered source")
    plot!(ax, target; label = "target")
    axislegend(ax)
    save("nrr-source-target.png", fig)
end
```

![](nrr-source-target.png)

Plotting the `nonrigid_displacement` reveals the tail movement:

```@setup 1
let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    hidedecorations!(ax)
    arrows2d!(ax, nonrigid_displacement; label = "displacement")
    axislegend(ax)
    save("nrr-displacement.png", fig)
end
```

![](nrr-displacement.png)
