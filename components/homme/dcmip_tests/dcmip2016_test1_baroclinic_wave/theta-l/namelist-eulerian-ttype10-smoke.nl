!
! theta-l_kokkos: DCMIP2016 test1 baroclinic wave — ne2 smoke test
!   ttype10 IMEX, Eulerian transport, qsplit=rsplit=1, native GLL output.
!   12 dyn steps -> 2 output frames. Validates the E10 tendency path with
!   the kokkos dycore end to end. Not a scientific configuration.
!
! Companion: namelist-eulerian-ttype10.nl for the ne=8 / ndays=9 run.
!_______________________________________________________________________
&ctl_nl
  nthreads          = -1
  partmethod        = 4
  topology          = "cube"
  test_case         = "dcmip2016_test1"
  ne                = 2
  qsize             = 6
  nmax              = 12                 ! 12 dyn steps -> 2 output frames
  statefreq         = 6
  restartfreq       = -1
  runtype           = 0
  tstep             = 300
  integration       = 'explicit'
  tstep_type        = 10                 ! ttype10_imex (theta-l_kokkos)
  rsplit            = 1
  qsplit            = 1
  transport_alg     = 0                  ! 0 = Eulerian (EulerStepFunctor)
  vert_remap_q_alg  = 1                  ! 1 = PPM_MIRRORED (kokkos accepts {1,3,10})
  nu                = 1e15
  nu_s              = 1e15
  nu_p              = 1e15
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
  output_prefix     = "hommexx-eulerian-ttype10-ne2-"
  output_dir        = "./movies/"
  output_timeunits  = 0,                 ! 0 = timesteps
  output_frequency  = 6                  ! every 6 dyn steps
  output_varnames1  = 'ps','T','Th','pnh','geo','u','v','w','omega','dp','Q','Q2','Q3','Q4','Q5','precl','zeta',
                      'FM_x','FM_y','FM_z','FT','FQ1','FQ2','FQ3'
  interp_type       = 0                  ! native GLL
  output_type       = 'netcdf4p'         ! HDF5-backed, required by native GLL path
  num_io_procs      = 1
/
&prof_inparm
  profile_outpe_num   = 100
  profile_single_file = .true.
/
