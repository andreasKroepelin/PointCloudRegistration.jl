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

First, we can find a *rigid transformation* that rotates and translates the
source such that they sit on top of each other:

```@repl 1
rigid_transformation = rigid_registration(source, target)
```

Now, `rigid_transformation(source)` matches `target` quite well:

```@repl 1
fig = Figure() # hide
ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true) # hide
hidedecorations!(ax) # hide
plot!(ax, rigid_transformation(source); label = "rigidly transformed source") # hide
plot!(ax, target; label = "target") # hide
axislegend(ax) # hide
save("rr-source-target.png", fig); # hide
```

![](rr-source-target.png)
