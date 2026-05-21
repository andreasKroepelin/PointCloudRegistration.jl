# PointCloudRegistration.jl

**PointCloudRegistration.jl** is a package for performing rigid and nonrigid
registration of point clouds, also known as *superimposing* or *aligning*.

## Introductory example

To understand what point cloud registration is and why we need it, let's have
a look at these two cute cats:

```@repl 1
using PointCloudRegistration
source, target = PointCloudRegistration.Assets.load_cats();
using GLMakie # hide
fig = Figure() # hide
ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true) # hide
hidedecorations!(ax) # hide
plot!(ax, source; label = "source") # hide
plot!(ax, target; label = "target") # hide
axislegend(ax) # hide
save("source-target.png", fig); # hide
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

```@repl 1
fig = Figure() # hide
ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true) # hide
hidedecorations!(ax) # hide
plot!(ax, source2; label = "rigidly registered source") # hide
plot!(ax, target; label = "target") # hide
axislegend(ax) # hide
save("rr-source-target.png", fig); # hide
```

![](rr-source-target.png)

Often, we are already done here.
However, we might be interested in explaining more of the differences between
the point clouds, for which we turn to _non-rigid registration_:

```@repl 1
cpd =  nonrigid_registration(
  source2,
  target,
  CoherentPointDrift(; corr_length = 50, expected_displacement = 100),
)
```

As we can see, this requires a bit more domain knowledge and fine-tuning of
parameters compared to rigid registration.
Applying the result to `source2` provides a new point cloud:

```@repl 1
source3 = cpd(source2)
```

The plot reveals the final result:

```@repl 1
fig = Figure() # hide
ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true) # hide
hidedecorations!(ax) # hide
plot!(ax, source3; label = "rigidly and non-rigidly registered source") # hide
plot!(ax, target; label = "target") # hide
axislegend(ax) # hide
save("nrr-source-target.png", fig); # hide
```

![](nrr-source-target.png)
