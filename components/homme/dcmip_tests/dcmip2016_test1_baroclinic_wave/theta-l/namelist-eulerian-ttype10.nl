!
! theta-l_kokkos: DCMIP2016 test1 baroclinic wave
!   ttype10 IMEX, Eulerian transport, qsplit=rsplit=1
!   NOTE: run with the theta-l_kokkos (C++) executable, e.g.
!         theta-l-nlev30-kokkos < namelist-eulerian-ttype10.nl
!_______________________________________________________________________
&ctl_nl
  nthreads          = -1
  partmethod        = 4
  topology          = "cube"
  test_case         = "dcmip2016_test1"
  ne                = 8
  qsize             = 6
  ndays             = 9
  statefreq         = 24
  restartfreq       = -1
  runtype           = 0
  tstep             = 300
  integration       = 'explicit'
  tstep_type        = 10                 ! ttype10_imex (theta-l_kokkos)
  rsplit            = 1
  qsplit            = 1
  transport_alg     = 0                  ! 0 = Eulerian (EulerStepFunctor)
  vert_remap_q_alg  = 1                  ! 1 = PPM_MIRRORED (kokkos accepts {1,3,10})
  nu                = 3e16
  nu_s              = 3e16
  nu_p              = 3e16
  nu_top            = 0
  limiter_option    = 9                  ! Eulerian requires 8 or 9
  hypervis_order    = 2
  hypervis_subcycle = 1
  moisture          = 'wet'
  theta_hydrostatic_mode = .false.
  theta_advect_form = 1
  dcmip16_prec_type = 1
  dcmip16_pbl_type  = -1
/
&vert_nl
  vfile_mid         = "../vcoord/camm-30.ascii"
  vfile_int         = "../vcoord/cami-30.ascii"
/
&analysis_nl
  output_prefix     = "hommexx-eulerian-ttype10-"
  output_dir        = "./movies/"
  output_timeunits  = 0,                 ! 0 = timesteps
  output_frequency  = 6                  ! every 6 dyn steps
  output_varnames1  = 'ps','T','Th','pnh','geo','u','v','w','omega','dp','Q','Q2','Q3','Q4','Q5','precl','zeta',
                      'FM_x','FM_y','FM_z','FT','FQ1','FQ2','FQ3'
  interp_type       = 0                  ! native GLL
  output_type       = 'netcdf4p'         ! HDF5-backed, required by native GLL path
  num_io_procs      = 16
/
&prof_inparm
  profile_outpe_num   = 100
  profile_single_file = .true.
/
