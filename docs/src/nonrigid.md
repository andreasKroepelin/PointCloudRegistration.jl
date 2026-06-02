# Non-rigid registration

While rigid registration asks how to rotate and translate the source to match
the target, _non-rigid_ registration tries to deform the source under certain
constraints to match the target.

```@docs
nonrigid_registration(source, target, algorithm)
```

## Displacement

In general, the result of a non-rigid registration can be expressed by a
mapping from the source points to the registered source points.

```@docs
PointCloudRegistration.Displacement
```

## Non-rigid registration algorithms

* [Coherent Point Drift](@ref)
* [Distance Preserving](@ref)
* Divergence Free
* [Optimal Transport](@ref)

### Coherent Point Drift
```@docs
CoherentPointDrift
nonrigid_registration(source, target, ::CoherentPointDrift)
prepare_source_coherentpointdrift
```

### Distance Preserving
```@docs
DistancePreserving
nonrigid_registration(source, target, ::DistancePreserving)
prepare_source_distancepreserving
```

### Optimal Transport
```@docs
EarthMover
nonrigid_registration(source, target, ::EarthMover)
```
