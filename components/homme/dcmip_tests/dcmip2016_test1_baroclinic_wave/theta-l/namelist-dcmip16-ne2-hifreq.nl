!
! theta-l: DCMIP2016 test 1 (moist baroclinic wave) — ne2 smoke test
!
! Purpose: end-to-end validation of the hifreq training-data pipeline (E9/E10)
!   at trivial cost before committing to an ne30 run. Two physics steps and out.
!   Not a scientific configuration — the wave does not develop at ne2.
!
! Companion: namelist-dcmip16-ne30-hifreq.nl for production training data.
!_______________________________________________________________________
&ctl_nl
  nthreads          = -1
  partmethod        = 4
  topology          = "cube"
  test_case         = "dcmip2016_test1"
  ne                = 2
  qsize             = 6
  nmax              = 12                           ! 2 physics steps (rsplit=6)
  statefreq         = 6
  restartfreq       = -1
  runtype           = 0
  tstep             = 300
  integration       = 'explicit'
  tstep_type        = 7
  rsplit            = 6
  qsplit            = 1
  nu                = 1e15
  nu_s              = 1e15
  nu_p              = 1e15
  nu_top            = 0
  limiter_option    = 9
  hypervis_order    = 2
  hypervis_subcycle = 1
  moisture          = 'wet'
  theta_hydrostatic_mode = .false.
  dcmip16_prec_type = 1
  dcmip16_pbl_type  = -1
/
&vert_nl
  vfile_mid         = "../vcoord/camm-30.ascii"
  vfile_int         = "../vcoord/cami-30.ascii"
/
&analysis_nl
  output_prefix     = "dcmip16-t1-ne2-hifreq-"
  output_dir        = "./movies/"
  output_timeunits  = 0
  output_frequency  = 6                            ! every physics step
  output_varnames1  = 'ps','T','Th','pnh','geo','u','v','w','dp','Q','Q2','Q3','Q4','Q5','precl','zeta',
                      'FM_x','FM_y','FM_z','FT','FQ1','FQ2','FQ3'
  interp_type       = 0                            ! native GLL
  output_type       = 'netcdf'
  num_io_procs      = 1
/
&prof_inparm
  profile_outpe_num   = 100
  profile_single_file = .true.
/
