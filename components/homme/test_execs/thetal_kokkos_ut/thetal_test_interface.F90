module thetal_test_interface

  use iso_c_binding,  only: c_int, c_bool, c_double, c_ptr, c_f_pointer
  use derivative_mod, only: derivative_t
  use element_mod,    only: element_t
  use kinds,          only: real_kind
  use hybvcoord_mod,  only: hvcoord_t
  use parallel_mod,   only: abortmp
  use geometry_mod,   only: set_area_correction_map0

  implicit none

  type(hvcoord_t) :: hvcoord
  type (derivative_t) :: deriv

  public :: init_f90
  public :: cleanup_f90
  public :: init_geo_views_f90
  public :: init_elements_geometry_f90
  public :: set_hvcoord_f90
  public :: initialize_reference_states_f90

contains

  subroutine init_f90 (ne, hyai, hybi, hyam, hybm, dvv, mp, ps0) bind(c)
    ! Create a cubed sphere mesh, with ne x ne elements per cube face
    use dimensions_mod, only: nlev, nlevp, np
    !
    ! Inputs
    !
    integer (kind=c_int), intent(in) :: ne
    real (kind=real_kind), intent(in) :: hyai(nlevp), hybi(nlevp), hyam(nlev), hybm(nlev)
    real (kind=real_kind), intent(in) :: ps0
    real (kind=real_kind), intent(out) :: dvv(np,np), mp(np,np)

    call init_mesh(.false., ne, ne, hyai, hybi, hyam, hybm, dvv, mp, ps0)
  end subroutine init_f90

  subroutine init_planar_f90 (ne_x, ne_y, Lx, Ly, Sx, Sy, hyai, hybi, hyam, hybm, dvv, mp, ps0) bind(c)
    ! Create a planar mesh, with ne_x x ne_y elements, covering the domain
    ! [Sx,Sx+Lx] x [Sy,Sy+Ly] (in meters)
    use dimensions_mod, only: nlev, nlevp, np
    !
    ! Inputs
    !
    integer (kind=c_int), intent(in) :: ne_x, ne_y
    real (kind=real_kind), intent(in) :: Lx, Ly, Sx, Sy
    real (kind=real_kind), intent(in) :: hyai(nlevp), hybi(nlevp), hyam(nlev), hybm(nlev)
    real (kind=real_kind), intent(in) :: ps0
    real (kind=real_kind), intent(out) :: dvv(np,np), mp(np,np)

    call init_mesh(.true., ne_x, ne_y, hyai, hybi, hyam, hybm, dvv, mp, ps0, [Lx, Ly, Sx, Sy])
  end subroutine init_planar_f90

  subroutine init_mesh (planar, ne_x_in, ne_y_in, hyai, hybi, hyam, hybm, dvv, mp, ps0, domain)
    ! Common part of init_f90 and init_planar_f90.
    ! The only differences between a cubed sphere and a planar mesh are in the settings
    ! that determine the topology/geometry (and the scale factors), and in the routines
    ! that set the element corners and the GLL points.
    use control_mod,            only: cubed_sphere_map, geometry, topology
    use cube_mod,               only: cube_init_atomic, set_corner_coordinates
    use planar_mod,             only: plane_init_atomic, plane_set_corner_coordinates
    use derivative_mod,         only: derivinit
    use dimensions_mod,         only: nelemd, nlev, nlevp, np, ne_x, ne_y
    use geometry_interface_mod, only: initmp_f90, init_cube_geometry_f90, init_connectivity_f90
    use geometry_interface_mod, only: par, elem
    use quadrature_mod,         only: gausslobatto, quadrature_t
    use physical_constants,     only: scale_factor, scale_factor_inv, laplacian_rigid_factor, &
                                      rearth, rrearth, Sx, Sy, Lx, Ly, dx, dy, dx_ref, dy_ref, domain_size
    !
    ! Inputs
    !
    logical, intent(in) :: planar
    ! For a cubed sphere, ne_x_in is ne (the number of elements per cube face edge), and ne_y_in is unused
    integer (kind=c_int), intent(in) :: ne_x_in, ne_y_in
    real (kind=real_kind), intent(in) :: hyai(nlevp), hybi(nlevp), hyam(nlev), hybm(nlev)
    real (kind=real_kind), intent(in) :: ps0
    real (kind=real_kind), intent(out) :: dvv(np,np), mp(np,np)
    ! Only used for planar meshes: Lx, Ly, Sx, Sy
    real (kind=real_kind), intent(in), optional :: domain(4)
    !
    ! Locals
    !
    integer :: ie
    type (quadrature_t) :: gp

    ! Set these explicitly, so that a cubed sphere mesh can be created
    ! after a planar one (and vice versa) in the same process
    if (planar) then
      topology = 'plane'
      geometry = 'plane'

      ne_x = ne_x_in
      ne_y = ne_y_in

      scale_factor = 1
      scale_factor_inv = 1
      laplacian_rigid_factor = 0

      Lx = domain(1)
      Ly = domain(2)
      Sx = domain(3)
      Sy = domain(4)

      domain_size = Lx * Ly
      dx = Lx/ne_x
      dy = Ly/ne_y
      dx_ref = 1.0D0/ne_x
      dy_ref = 1.0D0/ne_y
    else
      topology = 'cube'
      geometry = 'sphere'

      scale_factor = rearth
      scale_factor_inv = rrearth
      laplacian_rigid_factor = rrearth
    endif

    call derivinit(deriv)

    call initmp_f90()
    call init_cube_geometry_f90(ne_x_in) ! For planar meshes, the arg is unused (ne_x and ne_y are used)
    call init_connectivity_f90()

    gp=gausslobatto(np)  ! GLL points
    if (planar) then
      cubed_sphere_map = 2
      do ie=1,nelemd
        call plane_set_corner_coordinates(elem(ie))
      end do
      do ie=1,nelemd
        call plane_init_atomic(elem(ie),gp%points)
      enddo
    else
      cubed_sphere_map = 0
      do ie=1,nelemd
        call set_corner_coordinates(elem(ie))
      end do
      do ie=1,nelemd
        call cube_init_atomic(elem(ie),gp%points)
      enddo
    endif

    call init_common(hyai, hybi, hyam, hybm, dvv, mp, ps0)
  end subroutine init_mesh

  subroutine init_common(hyai, hybi, hyam, hybm, dvv, mp, ps0)
    use element_state,          only: allocate_element_arrays, setup_element_pointers_ie
    use mass_matrix_mod,        only: mass_matrix
    use dimensions_mod,         only: nelemd, nlev, nlevp, np
    use geometry_interface_mod, only: par, elem
    !
    ! Inputs
    !
    real (kind=real_kind), intent(in) :: hyai(nlevp), hybi(nlevp), hyam(nlev), hybm(nlev)
    real (kind=real_kind), intent(in) :: ps0
    real (kind=real_kind), intent(out) :: dvv(np,np), mp(np,np)
    !
    ! Locals
    !
    integer :: ie
    
    call allocate_element_arrays(nelemd)

    call mass_matrix(par,elem)

    ! Copy refFE matrices back to C
    mp = elem(1)%mp
    dvv = deriv%dvv

    do ie=1,nelemd
      call setup_element_pointers_ie(ie,elem(ie)%state, elem(ie)%derived, elem(ie)%accum)
    enddo

    call set_hvcoord_f90(hyai, hybi, hyam, hybm, ps0)

    deriv%dvv = dvv
  end subroutine init_common

  subroutine set_hvcoord_f90 (hyai, hybi, hyam, hybm, ps0) bind(c)
    use hybvcoord_mod,  only: set_layer_locations
    use dimensions_mod, only: nlev, nlevp
    !
    ! Inputs
    !
    real (kind=real_kind), intent(in) :: hyai(nlevp), hybi(nlevp), hyam(nlev), hybm(nlev)
    real (kind=real_kind), intent(in) :: ps0
    !
    ! Locals
    !
    integer :: k

    hvcoord%hyai = hyai
    hvcoord%hybi = hybi
    hvcoord%hyam = hyam
    hvcoord%hybm = hybm
    hvcoord%ps0 = ps0
    do k=1,nlev
      hvcoord%dp0(k) = (hvcoord%hyai(k+1) - hvcoord%hyai(k))*ps0 + &
                       (hvcoord%hybi(k+1) - hvcoord%hybi(k))*ps0
    enddo

    call set_layer_locations (hvcoord,.false.,.false.)
  end subroutine set_hvcoord_f90

  subroutine init_elements_geometry_f90 () bind(c)
    ! Pass the f90 geometry (and geopotential) of the mesh to the C++ Elements in the Context,
    ! using the same routines as the production code. Unlike init_geo_views_f90, this does not
    ! return the geometry to the caller: it requires the C++ Elements to be already
    ! created in the Context (e.g., by init_elements_c).
    use geometry_interface_mod, only: elem
    use prim_driver_mod,        only: prim_init_grid_views, prim_init_geopotential_views

    call prim_init_grid_views(elem)
    call prim_init_geopotential_views(elem)
  end subroutine init_elements_geometry_f90

  subroutine init_geo_views_f90 (d_ptr, dinv_ptr,        &
                       phis_ptr, gradphis_ptr, fcor_ptr, &
                       spmp_ptr, rspmp_ptr, tVisc_ptr,   &
                       sph2c_ptr,mdet_ptr,minv_ptr) bind(c)
    use dimensions_mod, only: nelemd, np
    use geometry_interface_mod, only: elem
    !
    ! Inputs
    !
    type (c_ptr), intent(in) :: d_ptr, dinv_ptr, spmp_ptr, rspmp_ptr, tVisc_ptr, fcor_ptr
    type (c_ptr), intent(in) :: sph2c_ptr, mdet_ptr, minv_ptr, phis_ptr, gradphis_ptr
    !
    ! Locals
    !
    integer :: ie
    real (kind=real_kind), pointer :: scalar2d (:,:,:)
    real (kind=real_kind), pointer :: vector2d (:,:,:,:)
    real (kind=real_kind), pointer :: tensor2d (:,:,:,:,:)

    ! Set all geometric views (we use 1 tensor, 1 vector, and 1 scalar temps)
    call c_f_pointer(d_ptr,    tensor2d, [np,np,2,2,nelemd])
    call c_f_pointer(mdet_ptr, scalar2d, [np,np,    nelemd])
    do ie=1,nelemd
      tensor2d(:,:,:,:,ie) = elem(ie)%d
      scalar2d(:,:,ie)     = elem(ie)%metdet
    enddo

    call c_f_pointer(dinv_ptr, tensor2d, [np,np,2,2,nelemd])
    call c_f_pointer(spmp_ptr, scalar2d, [np,np,    nelemd])
    do ie=1,nelemd
      tensor2d(:,:,:,:,ie) = elem(ie)%dinv
      scalar2d(:,:,ie)     = elem(ie)%spheremp
    enddo

    call c_f_pointer(minv_ptr,  tensor2d, [np,np,2,2,nelemd])
    call c_f_pointer(rspmp_ptr, scalar2d, [np,np,    nelemd])
    do ie=1,nelemd
      tensor2d(:,:,:,:,ie) = elem(ie)%metinv
      scalar2d(:,:,ie)     = elem(ie)%rspheremp
    enddo

    call c_f_pointer(sph2c_ptr, tensor2d, [np,np,3,2,nelemd])
    call c_f_pointer(mdet_ptr,  scalar2d, [np,np,    nelemd])
    do ie=1,nelemd
      tensor2d(:,:,:,:,ie) = elem(ie)%vec_sphere2cart
      scalar2d(:,:,ie)     = elem(ie)%metdet
    enddo

    call c_f_pointer(phis_ptr,     scalar2d, [np,np,    nelemd])
    call c_f_pointer(gradphis_ptr, vector2d, [np,np,2,  nelemd])
    call c_f_pointer(tVisc_ptr,    tensor2d, [np,np,2,2,nelemd])
    do ie=1,nelemd
      tensor2d(:,:,:,:,ie)      = elem(ie)%tensorVisc
      elem(ie)%derived%gradphis = vector2d(:,:,:,ie)
      elem(ie)%state%phis       = scalar2d(:,:,ie)
    enddo

    call c_f_pointer(fcor_ptr, scalar2d, [np, np, nelemd])
    do ie=1,nelemd
      scalar2d(:,:,ie) = elem(ie)%fcor
    enddo

  end subroutine init_geo_views_f90

  subroutine cleanup_f90 () bind(c)
    use edge_mod, only : FreeEdgeBuffer, edge_g
    use element_state, only : deallocate_element_arrays
    use geometry_interface_mod, only: cleanup_geometry_f90
    use parallel_mod, only: rrequest, srequest, status

    if (allocated(edge_g%buf)) then
      call FreeEdgeBuffer(edge_g)
    endif
    call cleanup_geometry_f90()

    ! Deallocate all module-scope variables, so the next iteration of catch2
    ! can successfully call init_f90 (which will try to allocate these again)
    deallocate(rrequest)
    deallocate(srequest)
    deallocate(status)

    call deallocate_element_arrays()

  end subroutine cleanup_f90


  subroutine initialize_reference_states_f90(phis_ptr, dp_ref_ptr, theta_ref_ptr, phi_ref_ptr) bind(c)
    use element_ops,    only: initialize_reference_states
    use dimensions_mod, only: nelemd, nlev, nlevp, np
    use theta_f2c_mod,  only : init_reference_states_c
    !
    ! Inputs
    !
    type (c_ptr), intent(in) :: phis_ptr, dp_ref_ptr, theta_ref_ptr, phi_ref_ptr
    !
    ! Locals
    !
    real (kind=real_kind), dimension(:,:,:),  pointer :: phis
    real (kind=real_kind), dimension(:,:,:,:),  pointer :: dp_ref, theta_ref, phi_ref
    integer :: ie

    call c_f_pointer(phis_ptr,      phis,      [np,np,      nelemd])
    call c_f_pointer(dp_ref_ptr,    dp_ref,    [np,np,nlev, nelemd])
    call c_f_pointer(theta_ref_ptr, theta_ref, [np,np,nlev, nelemd])
    call c_f_pointer(phi_ref_ptr,   phi_ref,   [np,np,nlevp,nelemd])

    ! Compute reference states
    do ie=1,nelemd
      call initialize_reference_states(hvcoord,             &
                                       phis(:,:,ie),        &
                                       dp_ref(:,:,:,ie),    &
                                       theta_ref(:,:,:,ie), &
                                       phi_ref(:,:,:,ie))
    end do

    ! Initialize reference states in C++
    call init_reference_states_c(dp_ref_ptr,theta_ref_ptr,phi_ref_ptr)

  end subroutine initialize_reference_states_f90

end module thetal_test_interface
