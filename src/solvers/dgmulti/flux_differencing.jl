# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

#   hadamard_sum!(du, A,
#                 flux_is_symmetric, volume_flux,
#                 orientation_or_normal_direction, u, equations)
#
# Computes the flux difference ∑_j A[i, j] * f(u_i, u_j) and accumulates the result into `du`.
# Called by `local_flux_differencing` to compute local contributions to flux differencing
# volume integrals.
#
# - `du`, `u` are vectors
# - `A` is the skew-symmetric flux differencing matrix
# - `flux_is_symmetric` is a `Val{<:Bool}` indicating if f(u_i, u_j) = f(u_j, u_i)
#
# The matrix `A` can be either dense or sparse. In the latter case, you should
# use the `adjoint` of a `SparseMatrixCSC` to mimic a `SparseMatrixCSR`, which
# is more efficient for matrix vector products.

# Version for dense operators and symmetric fluxes
@inline function hadamard_sum!(du, A,
                               flux_is_symmetric::Val{true}, volume_flux,
                               orientation_or_normal_direction, u, equations)
  row_ids, col_ids = axes(A)

  for i in row_ids
    u_i = u[i]
    du_i = du[i]
    for j in col_ids
      # This routine computes only the upper-triangular part of the hadamard sum (A .* F).
      # We avoid computing the lower-triangular part, and instead accumulate those contributions
      # while computing the upper-triangular part (using the fact that A is skew-symmetric and F
      # is symmetric).
      if j > i
        u_j = u[j]
        AF_ij = A[i,j] * volume_flux(u_i, u_j, orientation_or_normal_direction, equations)
        du_i = du_i + AF_ij
        du[j] = du[j] - AF_ij
      end
    end
    du[i] = du_i
  end
end

# Version for dense operators and non-symmetric fluxes
@inline function hadamard_sum!(du, A,
                               flux_is_symmetric::Val{false}, volume_flux,
                               orientation::Integer, u, equations)
  row_ids, col_ids = axes(A)

  for i in row_ids
    u_i = u[i]
    du_i = du[i]
    for j in col_ids
      u_j = u[j]
      f_ij = volume_flux(u_i, u_j, orientation, equations)
      du_i = du_i + A[i,j] * f_ij
    end
    du[i] = du_i
  end
end

# TODO: DGMulti. Fix for curved meshes.
@inline function hadamard_sum!(du, A,
                               flux_is_symmetric::Val{false}, volume_flux,
                               normal_direction::AbstractVector, u, equations)
  row_ids, col_ids = axes(A)

  for i in row_ids
    u_i = u[i]
    du_i = du[i]
    for j in col_ids
      u_j = u[j]
      # The `normal_direction::AbstractVector` has to be passed in twice.
      # This is because on curved meshes, nonconservative fluxes are
      # evaluated using both the normal and its average at interfaces.
      f_ij = volume_flux(u_i, u_j, normal_direction, normal_direction, equations)
      du_i = du_i + A[i,j] * f_ij
    end
    du[i] = du_i
  end
end

# Version for sparse operators and symmetric fluxes
@inline function hadamard_sum!(du, A::LinearAlgebra.Adjoint{<:Any, <:AbstractSparseMatrixCSC},
                               flux_is_symmetric::Val{true}, volume_flux,
                               orientation_or_normal_direction, u, equations)
  A_base = parent(A) # the adjoint of a SparseMatrixCSC is basically a SparseMatrixCSR
  row_ids = axes(A, 2)
  rows = rowvals(A_base)
  vals = nonzeros(A_base)

  for i in row_ids
    u_i = u[i]
    du_i = du[i]
    for id in nzrange(A_base, i)
      j = rows[id]
      # This routine computes only the upper-triangular part of the hadamard sum (A .* F).
      # We avoid computing the lower-triangular part, and instead accumulate those contributions
      # while computing the upper-triangular part (using the fact that A is skew-symmetric and F
      # is symmetric).
      if j > i
        u_j = u[j]
        A_ij = vals[id]
        AF_ij = A_ij * volume_flux(u_i, u_j, orientation_or_normal_direction, equations)
        du_i = du_i + AF_ij
        du[j] = du[j] - AF_ij
      end
    end
    du[i] = du_i
  end
end

# TODO: DGMulti. Fix for curved meshes.
# Version for sparse operators and non-symmetric fluxes
@inline function hadamard_sum!(du, A::LinearAlgebra.Adjoint{<:Any, <:AbstractSparseMatrixCSC},
                               flux_is_symmetric::Val{false}, volume_flux,
                               normal_direction::AbstractVector, u, equations)
  A_base = parent(A) # the adjoint of a SparseMatrixCSC is basically a SparseMatrixCSR
  row_ids = axes(A, 2)
  rows = rowvals(A_base)
  vals = nonzeros(A_base)

  for i in row_ids
    u_i = u[i]
    du_i = du[i]
    for id in nzrange(A_base, i)
      j = rows[id]
      # The `normal_direction::AbstractVector` has to be passed in twice.
      # This is because on curved meshes, nonconservative fluxes are
      # evaluated using both the normal and its average at interfaces.
      u_j = u[j]
      f_ij = volume_flux(u_i, u_j, normal_direction, normal_direction, equations)
      du_i = du_i + A[i,j] * f_ij
    end
    du[i] = du_i
  end
end


# For DGMulti implementations, we construct "physical" differentiation operators by taking linear
# combinations of reference differentiation operators scaled by geometric change of variables terms.
# We use a lazy evaluation of physical differentiation operators, so that we can compute linear
# combinations of differentiation operators on-the-fly in an allocation-free manner.
@inline function build_lazy_physical_derivative(element, orientation,
                                                mesh::VertexMappedMesh{2}, dg, cache,
                                                operator_scaling = 1.0)
  @unpack Qrst_skew = cache
  @unpack rxJ, sxJ, ryJ, syJ = mesh.md
  scaling = 2 * operator_scaling
  if orientation == 1
    return LazyMatrixLinearCombo(Qrst_skew, scaling .* (rxJ[1,element], sxJ[1,element]))
  else # if orientation == 2
    return LazyMatrixLinearCombo(Qrst_skew, scaling .* (ryJ[1,element], syJ[1,element]))
  end
end

@inline function build_lazy_physical_derivative(element, orientation,
                                                mesh::VertexMappedMesh{3}, dg, cache,
                                                operator_scaling = 1.0)
  @unpack Qrst_skew = cache
  @unpack rxJ, sxJ, txJ, ryJ, syJ, tyJ, rzJ, szJ, tzJ = mesh.md
  scaling = 2 * operator_scaling
  if orientation == 1
    return LazyMatrixLinearCombo(Qrst_skew, scaling .* (rxJ[1,element], sxJ[1,element], txJ[1,element]))
  elseif orientation == 2
    return LazyMatrixLinearCombo(Qrst_skew, scaling .* (ryJ[1,element], syJ[1,element], tyJ[1,element]))
  else # if orientation == 3
    return LazyMatrixLinearCombo(Qrst_skew, scaling .* (rzJ[1,element], szJ[1,element], tzJ[1,element]))
  end
end

# Return the contravariant basis vector corresponding to the Cartesian
# coordinate diretion `orientation` in a given `element` of the `mesh`.
# The contravariant basis vectors have entries `dx_i / dxhat_j` where
# j ∈ {1, ..., NDIMS}. Here, `x_i` and `xhat_j` are the ith physical coordinate
# and jth reference coordinate, respectively. These are geometric terms which
# appear when using the chain rule to compute physical derivatives as a linear
# combination of reference derivatives.
@inline function get_contravariant_vector(element, orientation, mesh::VertexMappedMesh{1})
  @unpack rxJ = mesh.md
  return 2 * SVector(rxJ[1, element]) # the 1D contravariant vector reduces to a scaling.
end

@inline function get_contravariant_vector(element, orientation, mesh::VertexMappedMesh{2})
  @unpack rxJ, sxJ, ryJ, syJ = mesh.md
  if orientation == 1
    return 2 * SVector(rxJ[1, element], ryJ[1, element])
  else # if orientation == 2
    return 2 * SVector(sxJ[1, element], syJ[1, element])
  end
end

@inline function get_contravariant_vector(element, orientation, mesh::VertexMappedMesh{3})
  @unpack rxJ, sxJ, txJ, ryJ, syJ, tyJ, rzJ, szJ, tzJ = mesh.md
  if orientation == 1
    return 2 * SVector(rxJ[1, element], ryJ[1, element], rzJ[1, element])
  elseif orientation == 2
    return 2 * SVector(sxJ[1, element], syJ[1, element], szJ[1, element])
  else # if orientation == 3
    return 2 * SVector(txJ[1, element], tyJ[1, element], tzJ[1, element])
  end
end

# use hybridized SBP operators for general flux differencing schemes.
function compute_flux_differencing_SBP_matrices(dg::DGMulti)
  compute_flux_differencing_SBP_matrices(dg, has_sparse_operators(dg))
end

function compute_flux_differencing_SBP_matrices(dg::DGMulti, sparse_operators)
  rd = dg.basis
  Qrst_hybridized, VhP, Ph = StartUpDG.hybridized_SBP_operators(rd)
  Qrst_skew = map(A -> 0.5 * (A - A'), Qrst_hybridized)
  if sparse_operators isa Val{true}
    Qrst_skew = map(Qi -> droptol!(sparse(Qi'), 100 * eps(eltype(Qi)))', Qrst_skew)
  end
  return Qrst_skew, VhP, Ph
end

# use traditional multidimensional SBP operators for SBP approximation types.
function compute_flux_differencing_SBP_matrices(dg::DGMultiFluxDiff{<:SBP}, sparse_operators)
  rd = dg.basis
  @unpack M, Drst, Pq = rd
  Qrst = map(D -> M * D, Drst)
  Qrst_skew = map(A -> 0.5 * (A - A'), Qrst)
  if sparse_operators isa Val{true}
    Qrst_skew = map(Qi -> droptol!(sparse(Qi'), 100 * eps(eltype(Qi)))', Qrst_skew)
  end
  return Qrst_skew
end


# For flux differencing SBP-type approximations, store solutions in Matrix{SVector{nvars}}.
# This results in a slight speedup for `calc_volume_integral!`.
function allocate_nested_array(uEltype, nvars, array_dimensions, dg::DGMultiFluxDiff{<:SBP})
  return zeros(SVector{nvars, uEltype}, array_dimensions...)
end

function create_cache(mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:SBP}, RealT, uEltype)

  rd = dg.basis
  md = mesh.md

  # for use with flux differencing schemes
  Qrst_skew = compute_flux_differencing_SBP_matrices(dg)

  # Todo: DGMulti. Factor common storage into a struct (MeshDataCache?) for reuse across solvers?
  # storage for volume quadrature values, face quadrature values, flux values
  nvars = nvariables(equations)
  u_values = allocate_nested_array(uEltype, nvars, size(md.xq), dg)
  u_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)
  flux_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)
  lift_scalings = rd.wf ./ rd.wq[rd.Fmask] # lift scalings for diag-norm SBP operators

  local_values_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nq,), dg) for _ in 1:Threads.nthreads()]

  # Use an array of SVectors (chunks of `nvars` are contiguous in memory) to speed up flux differencing
  fluxdiff_local_threaded = [zeros(SVector{nvars, uEltype}, rd.Nq) for _ in 1:Threads.nthreads()]

  return (; md, Qrst_skew,
            invJ = inv.(md.J), lift_scalings, inv_wq = inv.(rd.wq),
            u_values, u_face_values, flux_face_values,
            local_values_threaded, fluxdiff_local_threaded)
end

# most general create_cache: works for `DGMultiFluxDiff{<:Polynomial}`
function create_cache(mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff, RealT, uEltype)

  rd = dg.basis
  @unpack md = mesh

  Qrst_skew, VhP, Ph = compute_flux_differencing_SBP_matrices(dg)

  # temp storage for entropy variables at volume quad points
  nvars = nvariables(equations)
  entropy_var_values = allocate_nested_array(uEltype, nvars, (rd.Nq, md.num_elements), dg)

  # storage for all quadrature points (concatenated volume / face quadrature points)
  num_quad_points_total = rd.Nq + rd.Nfq
  entropy_projected_u_values = allocate_nested_array(uEltype, nvars, (num_quad_points_total, md.num_elements), dg)
  projected_entropy_var_values = allocate_nested_array(uEltype, nvars, (num_quad_points_total, md.num_elements), dg)

  # For this specific solver, `prolong2interfaces` will not be used anymore.
  # Instead, this step is also performed in `entropy_projection!`. Thus, we set
  # `u_face_values` as a `view` into `entropy_projected_u_values`. We do not do
  # the same for `u_values` since we will use that with LoopVectorization, which
  # cannot handle such views as of v0.12.66, the latest version at the time of writing.
  u_values = allocate_nested_array(uEltype, nvars, size(md.xq), dg)
  u_face_values = view(entropy_projected_u_values, rd.Nq+1:num_quad_points_total, :)
  flux_face_values = similar(u_face_values)

  # local storage for interface fluxes, rhs, and source
  local_values_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nq,), dg) for _ in 1:Threads.nthreads()]

  # Use an array of SVectors (chunks of `nvars` are contiguous in memory) to speed up flux differencing
  # The result is then transferred to rhs_local_threaded::StructArray{<:SVector} before
  # projecting it and storing it into `du`.
  fluxdiff_local_threaded = [zeros(SVector{nvars, uEltype}, num_quad_points_total) for _ in 1:Threads.nthreads()]
  rhs_local_threaded = [allocate_nested_array(uEltype, nvars, (num_quad_points_total,), dg)  for _ in 1:Threads.nthreads()]

  return (; md, Qrst_skew,
            VhP, Ph, invJ = inv.(md.J),
            entropy_var_values, projected_entropy_var_values, entropy_projected_u_values,
            u_values, u_face_values,  flux_face_values,
            local_values_threaded, fluxdiff_local_threaded, rhs_local_threaded)
end

# TODO: DGMulti. Address hard-coding of `entropy2cons!` and `cons2entropy!` for this function.
function entropy_projection!(cache, u, mesh::VertexMappedMesh, equations, dg::DGMulti)

  rd = dg.basis
  @unpack Vq = rd
  @unpack VhP, entropy_var_values, u_values = cache
  @unpack projected_entropy_var_values, entropy_projected_u_values = cache

  apply_to_each_field(mul_by!(Vq), u_values, u)

  cons2entropy!(entropy_var_values, u_values, equations)

  # "VhP" fuses the projection "P" with interpolation to volume and face quadrature "Vh"
  apply_to_each_field(mul_by!(VhP), projected_entropy_var_values, entropy_var_values)

  entropy2cons!(entropy_projected_u_values, projected_entropy_var_values, equations)
  return nothing
end

@inline function cons2entropy!(entropy_var_values::StructArray,
                               u_values          ::StructArray,
                               equations)
  @threaded for i in eachindex(u_values)
    entropy_var_values[i] = cons2entropy(u_values[i], equations)
  end
end

@inline function entropy2cons!(entropy_projected_u_values  ::StructArray,
                               projected_entropy_var_values::StructArray,
                               equations)
  @threaded for i in eachindex(projected_entropy_var_values)
    entropy_projected_u_values[i] = entropy2cons(projected_entropy_var_values[i], equations)
  end
end


# Trait-like system to dispatch based on whether or not the SBP operators are sparse.
# Designed to be extendable to include specialized `approximation_types` too.
@inline function has_sparse_operators(dg::DGMultiFluxDiff)
  rd = dg.basis
  return has_sparse_operators(rd.elementType, rd.approximationType)
end
# For traditional SBP operators on triangles, the operators are fully dense. We avoid using
# sum factorization here, which is slower for fully dense matrices.
@inline has_sparse_operators(::Union{Tri, Tet}, approx_type::AT) where {AT <: SBP} = Val{false}()

# Polynomial-based solvers use hybridized SBP operators, which have blocks scaled by outward
# normal components. This implies that operators for different coordinate directions have
# different sparsity patterns. We default to using sum factorization (which is faster when
# operators are sparse) for all `<:Polynomial` approximation types.
@inline has_sparse_operators(element_type, approx_type::Polynomial) = Val{true}()

# SBP/GaussSBP operators on quads/hexes use tensor-product operators. Thus, sum factorization is
# more efficient and we use the sparsity structure.
@inline has_sparse_operators(::Union{Quad, Hex}, approx_type::AT) where {AT <: SBP} = Val{true}()
@inline has_sparse_operators(::Union{Quad, Hex}, approx_type::GaussSBP) = Val{true}()

# Todo: DGMulti. Dispatch on curved/non-curved mesh types, this code only works for affine meshes (accessing rxJ[1,e],...)
# Computes flux differencing contribution from each Cartesian direction over a single element.
# For dense operators, we do not use sum factorization.
@inline function local_flux_differencing!(fluxdiff_local, u_local, element_index,
                                          has_nonconservative_terms::Val{false}, volume_integral,
                                          has_sparse_operators::Val{false}, mesh,
                                          equations, dg, cache) where {Flux}
  @unpack volume_flux = volume_integral
  for dim in eachdim(mesh)
    Qi_skew = build_lazy_physical_derivative(element_index, dim, mesh, dg, cache)
    # Val{true}() indicates the volume flux is symmetric
    hadamard_sum!(fluxdiff_local, Qi_skew,
                  Val{true}(), volume_flux,
                  dim, u_local, equations)
  end
end

@inline function local_flux_differencing!(fluxdiff_local, u_local, element_index,
                                          has_nonconservative_terms::Val{true}, volume_integral,
                                          has_sparse_operators::Val{false}, mesh,
                                          equations, dg, cache) where {Flux}
  flux_conservative, flux_nonconservative = volume_integral.volume_flux
  for dim in eachdim(mesh)
    Qi_skew = build_lazy_physical_derivative(element_index, dim, mesh, dg, cache)
    # Val{true}() indicates the flux is symmetric.
    hadamard_sum!(fluxdiff_local, Qi_skew,
                  Val{true}(), flux_conservative,
                  dim, u_local, equations)

    # The final argument .5 scales the operator by 1/2 for the nonconservative terms.
    half_Qi_skew = build_lazy_physical_derivative(element_index, dim, mesh, dg, cache, 0.5)
    # Val{false}() indicates the flux is non-symmetric.
    hadamard_sum!(fluxdiff_local, half_Qi_skew,
                  Val{false}(), flux_nonconservative,
                  dim, u_local, equations)
  end
end

# Todo: DGMulti. Dispatch on curved/non-curved mesh types, this code only works for affine meshes (accessing rxJ[1,e],...)
# When the operators are sparse, we use the sum-factorization approach to
# computing flux differencing.
@inline function local_flux_differencing!(fluxdiff_local, u_local, element_index,
                                          has_nonconservative_terms::Val{false}, volume_integral,
                                          has_sparse_operators::Val{true}, mesh,
                                          equations, dg, cache)
  @unpack Qrst_skew = cache
  @unpack volume_flux = volume_integral
  for dim in eachdim(mesh)
    # There are two ways to write this flux differencing discretization on affine meshes.
    #
    # 1. Use numerical fluxes in Cartesian directions and sum up the discrete derivative
    #    operators per coordinate direction accordingly.
    # 2. Use discrete derivative operators per coordinate direction and corresponding
    #    numerical fluxes in arbitrary (non-Cartesian) space directions.
    #
    # The first option makes it necessary to sum up the individual sparsity
    # patterns of each reference coordinate direction. On tensor-product
    # elements such as `Quad()` or `Hex()` elements, this increases the number of
    # potentially expensive numerical flux evaluations by a factor of `ndims(mesh)`.
    # Thus, we use the second option below (which basically corresponds to the
    # well-known sum factorization on tensor product elements).
    # Note that there is basically no difference for dense derivative operators.
    normal_direction = get_contravariant_vector(element_index, dim, mesh)
    Q_skew = Qrst_skew[dim]

    # Val{true}() indicates the flux is symmetric
    hadamard_sum!(fluxdiff_local, Q_skew,
                  Val{true}(), volume_flux,
                  normal_direction, u_local, equations)
  end
end

@inline function local_flux_differencing!(fluxdiff_local, u_local, element_index,
                                          has_nonconservative_terms::Val{true}, volume_integral,
                                          has_sparse_operators::Val{true}, mesh,
                                          equations, dg, cache)
  @unpack Qrst_skew = cache
  flux_conservative, flux_nonconservative = volume_integral.volume_flux
  for dim in eachdim(mesh)
    normal_direction = get_contravariant_vector(element_index, dim, mesh)
    Q_skew = Qrst_skew[dim]

    # Val{true}() indicates the flux is symmetric
    hadamard_sum!(fluxdiff_local, Q_skew,
                  Val{true}(), flux_conservative,
                  normal_direction, u_local, equations)

    # We scale the operator by 1/2 for the nonconservative terms.
    half_Q_skew = LazyMatrixLinearCombo((Q_skew, ), (0.5, ))
    # Val{false}() indicates the flux is non-symmetric
    hadamard_sum!(fluxdiff_local, half_Q_skew,
                  Val{false}(), flux_nonconservative,
                  normal_direction, u_local, equations)
  end
end


function calc_volume_integral!(du, u, mesh::VertexMappedMesh,
                               have_nonconservative_terms, equations,
                               volume_integral, dg::DGMultiFluxDiff{<:Polynomial},
                               cache)

  @unpack entropy_projected_u_values, Ph = cache
  @unpack fluxdiff_local_threaded, rhs_local_threaded = cache

  @threaded for e in eachelement(mesh, dg, cache)
    fluxdiff_local = fluxdiff_local_threaded[Threads.threadid()]
    fill!(fluxdiff_local, zero(eltype(fluxdiff_local)))
    u_local = view(entropy_projected_u_values, :, e)

    local_flux_differencing!(fluxdiff_local, u_local, e,
                             have_nonconservative_terms, volume_integral,
                             has_sparse_operators(dg),
                             mesh, equations, dg, cache)

    # convert fluxdiff_local::Vector{<:SVector} to StructArray{<:SVector} for faster
    # apply_to_each_field performance.
    rhs_local = rhs_local_threaded[Threads.threadid()]
    for i in Base.OneTo(length(fluxdiff_local))
      rhs_local[i] = fluxdiff_local[i]
    end
    apply_to_each_field(mul_by_accum!(Ph), view(du, :, e), rhs_local)
  end
end

function calc_volume_integral!(du, u, mesh::VertexMappedMesh,
                               have_nonconservative_terms, equations,
                               volume_integral, dg::DGMultiFluxDiff{<:SBP},
                               cache)

  @unpack fluxdiff_local_threaded, inv_wq = cache

  @threaded for e in eachelement(mesh, dg, cache)
    fluxdiff_local = fluxdiff_local_threaded[Threads.threadid()]
    fill!(fluxdiff_local, zero(eltype(fluxdiff_local)))
    u_local = view(u, :, e)

    local_flux_differencing!(fluxdiff_local, u_local, e,
                             have_nonconservative_terms, volume_integral,
                             has_sparse_operators(dg),
                             mesh, equations, dg, cache)

    for i in each_quad_node(mesh, dg, cache)
      du[i, e] = du[i, e] + fluxdiff_local[i] * inv_wq[i]
    end
  end
end


# Specialize since `u_values` isn't computed for DGMultiFluxDiff{<:SBP} solvers.
function calc_sources!(du, u, t, source_terms,
                       mesh, equations, dg::DGMultiFluxDiff{<:SBP}, cache)
  md = mesh.md

  @threaded for e in eachelement(mesh, dg, cache)
    for i in each_quad_node(mesh, dg, cache)
      du[i, e] += source_terms(u[i, e], SVector(getindex.(md.xyzq, i, e)), t, equations)
    end
  end
end


# Specializes on Polynomial (e.g., modal) DG methods with a flux differencing volume integral, e.g.,
# an entropy conservative/stable discretization. For modal DG schemes, an extra `entropy_projection!`
# is required (see https://doi.org/10.1016/j.jcp.2018.02.033, Section 4.3).
function rhs!(du, u, t, mesh, equations, initial_condition, boundary_conditions::BC,
              source_terms::Source, dg::DGMultiFluxDiff{<:Union{Polynomial, GaussSBP}}, cache) where {Source, BC}

  @trixi_timeit timer() "reset ∂u/∂t" fill!(du, zero(eltype(du)))

  # this function evaluates the solution at volume and face quadrature points (which was previously
  # done in `prolong2interfaces` and `calc_volume_integral`)
  @trixi_timeit timer() "entropy_projection!" entropy_projection!(cache, u, mesh, equations, dg)

  @trixi_timeit timer() "volume integral" calc_volume_integral!(
    du, u, mesh, have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  # the following functions are the same as in VolumeIntegralWeakForm, and can be reused from dg.jl
  @trixi_timeit timer() "interface flux" calc_interface_flux!(cache, dg.surface_integral, mesh,
                                                              have_nonconservative_terms(equations),
                                                              equations, dg)

  @trixi_timeit timer() "boundary flux" calc_boundary_flux!(cache, t, boundary_conditions,
                                                            mesh, equations, dg)

  @trixi_timeit timer() "surface integral" calc_surface_integral!(du, u, dg.surface_integral,
                                                                  mesh, equations, dg, cache)

  @trixi_timeit timer() "Jacobian" invert_jacobian!(du, mesh, equations, dg, cache)

  @trixi_timeit timer() "sources terms" calc_sources!(du, u, t, source_terms,
                                                     mesh, equations, dg, cache)

  return nothing
end


end # @muladd
