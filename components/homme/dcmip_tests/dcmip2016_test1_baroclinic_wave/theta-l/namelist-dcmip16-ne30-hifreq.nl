!
! theta-l: DCMIP2016 test 1 (moist baroclinic wave) — high-frequency training-data run
!
! Purpose: generate NN training data for pyhommexx D8 bridging experiment.
!   Ground-truth physics tendencies (FM_x, FM_y, FT, FQ{1,2,3}) are emitted
!   at physics-step cadence alongside prognostic state. Native GLL output
!   (interp_type=0) so the training data matches the grid pyhommexx sees.
!
! Volume: ne=30, nlev=30, ndays=60, output every physics step (rsplit=6 * tstep=300s = 1800s)
!   → 2880 snapshots × ~20 fields × ~20 MB/field ≈ ~1.2 TB. Trim ndays or
!   output_varnames1 if that's tight on scratch. Consider stride subsampling in
!   the pyhomme/dcmip16_data.TendencyDataset side rather than dropping fields
!   at write time.
!
! Companion: namelist-dcmip16-ne2-hifreq.nl for smoke testing the pipeline end to end.
!_______________________________________________________________________
&ctl_nl
  nthreads          = -1
  partmethod        = 4
  topology          = "cube"
  test_case         = "dcmip2016_test1"
  ne                = 30
  qsize             = 6
  ndays             = 60
  statefreq         = 72
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
  output_prefix     = "dcmip16-t1-ne30-hifreq-"
  output_dir        = "./movies/"
  output_timeunits  = 0                            ! 0=timesteps (dyn tsteps)
  output_frequency  = 6                            ! every rsplit dyn steps = every physics step
  ! Prognostic state (NN inputs) + ground-truth physics tendencies (NN targets, E10).
  output_varnames1  = 'ps','T','Th','pnh','geo','u','v','w','dp','Q','Q2','Q3','Q4','Q5','precl','zeta',
                      'FM_x','FM_y','FM_z','FT','FQ1','FQ2','FQ3'
  interp_type       = 0                            ! native GLL — no interp before NN sees it
  output_type       = 'netcdf'
  num_io_procs      = 16
/
&prof_inparm
  profile_outpe_num   = 100
  profile_single_file = .true.
/
