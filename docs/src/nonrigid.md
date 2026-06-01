# Non-rigid registration

While rigid registration asks how to rotate and translate the source to match
the target, _non-rigid_ registration tries to deform the source under certain
constraints to match the target.

```@docs
nonrigid_registration(source, target, algorithm)
```

## Non-rigid registration algorithms

* [Coherent Point Drift](@ref)
* Distance Preserving
* Divergence Free
* Optimal Transport

### Coherent Point Drift
```@docs
CoherentPointDrift
nonrigid_registration(source, target, ::CoherentPointDrift)
prepare_source_coherentpointdrift
```
