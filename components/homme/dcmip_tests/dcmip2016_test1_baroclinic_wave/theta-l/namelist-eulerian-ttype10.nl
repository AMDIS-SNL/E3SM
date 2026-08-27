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
  output_dir        = "./movies/"
  output_timeunits  = 0,                 ! 0 = timesteps
  output_frequency  = 1                  ! every dynamics step
  output_varnames1  = 'T','ps','pnh','geo','u','v','w','omega','Th','Q','Q2','Q3','Q4','Q5','precl','zeta'
  interp_type       = 1
  output_type       = 'netcdf'
  num_io_procs      = 16
  interp_gridtype   = 1
/
&prof_inparm
  profile_outpe_num   = 100
  profile_single_file = .true.
/
