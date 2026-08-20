#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

module common_movie_mod
  use control_mod, only : test_case, max_string_len

#ifndef HOMME_WITHOUT_PIOLIBRARY
  use common_io_mod, only : output_start_time, output_end_time, &
       max_output_streams, output_frequency, nf_double, nf_int, &
       max_output_variables
#else
  use common_io_mod, only : output_start_time, output_end_time, &
       max_output_streams, output_frequency,                    &
       max_output_variables
#endif
  implicit none
  private

#ifndef HOMME_WITHOUT_PIOLIBRARY
  public ::  varrequired, vartype, varnames, varcnt, vardims, &
             dimnames, maxdims
#endif

  public :: nextoutputstep, setvarnames

#ifndef HOMME_WITHOUT_PIOLIBRARY

#ifdef _PRIM
#ifdef HOMME_AMDIS_PROJECT
  ! +10 vars: dp, Q5, Q6, FM_x, FM_y, FM_z, FT, FQ1, FQ2, FQ3 (E10 training-data diagnostics)
  integer, parameter :: varcnt = 42 + 10
#else
  integer, parameter :: varcnt =  42
#endif

  integer, parameter :: maxdims =  7

  character*(*), parameter :: varnames(varcnt)=(/'ps         ', &
                                                 'geos       ', &
                                                 'PHIS       ', &
                                                 'precl      ', &
                                                 'area       ', &
                                                 'cv_lat     ', &
                                                 'cv_lon     ', &
                                                 'corners    ', &
                                                 'faceno     ', &
                                                 'zeta       ', &
                                                 'div        ', &
                                                 'T          ', &
                                                 'Th         ', &
                                                 'u          ', &
                                                 'v          ', &
                                                 'w          ', &
                                                 'w_i        ', &
                                                 'mu_i       ', &
                                                 'geo_i      ', &
                                                 'pnh        ', &
                                                 'ke         ', &
                                                 'hypervis   ', &
                                                 'Q          ', &
                                                 'Q2         ', &
                                                 'Q3         ', &
                                                 'Q4         ', &
                                                 'geo        ', &
                                                 'omega      ', &
                                                 'dp3d       ', &
                                                 'lat        ', &
                                                 'lon        ', &
                                                 'lev        ', &
                                                 'ilev       ', &
                                                 'hyam       ', &
                                                 'hybm       ', &
                                                 'hyai       ', &
                                                 'hybi       ', &
                                                 'Th_sens    ', &
                                                 'u_sens     ', &
                                                 'v_sens     ', &
                                                 'qv_sens    ', &
                                                 'time       '  &
#ifdef HOMME_AMDIS_PROJECT
                                                ,'dp         ', &
                                                 'Q5         ', &
                                                 'Q6         ', &
                                                 'FM_x       ', &
                                                 'FM_y       ', &
                                                 'FM_z       ', &
                                                 'FT         ', &
                                                 'FQ1        ', &
                                                 'FQ2        ', &
                                                 'FQ3        '  &
#endif
                                                 /)


  integer, parameter :: vardims(maxdims,varcnt) =  reshape( (/ 1,5,0,0,0,0,0, & ! ps
                                                               1,0,0,0,0,0,0, & ! geos
                                                               1,0,0,0,0,0,0, & ! PHIS=geos
                                                               1,5,0,0,0,0,0, & ! precl
                                                               1,0,0,0,0,0,0, & ! area
                                                               1,2,0,0,0,0,0, & ! cv_lat
                                                               1,2,0,0,0,0,0, & ! cv_lon
                                                               6,2,0,0,0,0,0, & ! subelement_corners
                                                               1,2,5,0,0,0,0, & ! faceno
                                                               1,2,5,0,0,0,0, & ! zeta
                                                               1,2,5,0,0,0,0, & ! div
                                                               1,2,5,0,0,0,0, & ! T
                                                               1,2,5,0,0,0,0, & ! Th
                                                               1,2,5,0,0,0,0, & ! u
                                                               1,2,5,0,0,0,0, & ! v
                                                               1,2,5,0,0,0,0, & ! w
                                                               1,3,5,0,0,0,0, & ! w_i
                                                               1,3,5,0,0,0,0, & ! mu_i
                                                               1,3,5,0,0,0,0, & ! geo_i
                                                               1,2,5,0,0,0,0, & ! pnh
                                                               1,2,5,0,0,0,0, & ! ke
                                                               1,5,0,0,0,0,0, & ! hypervis
                                                               1,2,5,0,0,0,0, & ! Q
                                                               1,2,5,0,0,0,0, & ! Q2
                                                               1,2,5,0,0,0,0, & ! Q3
                                                               1,2,5,0,0,0,0, & ! Q4
                                                               1,2,5,0,0,0,0, & ! geo
                                                               1,2,5,0,0,0,0, & !omega
                                                               1,2,5,0,0,0,0, & !dp3d
                                                               1,0,0,0,0,0,0, &  ! lat (y for planar)
                                                               1,0,0,0,0,0,0, &  ! lon (x for planar)
                                                               2,0,0,0,0,0,0, &  ! lev
                                                               3,0,0,0,0,0,0, &  ! ilev
                                                               2,0,0,0,0,0,0, &  !hyam
                                                               2,0,0,0,0,0,0, &  !hybm
                                                               3,0,0,0,0,0,0, &  !hyai
                                                               3,0,0,0,0,0,0, &  !hybi
                                                               1,2,7,5,0,0,0, &  !Th_sens
                                                               1,2,7,5,0,0,0, &  !u_sens
                                                               1,2,7,5,0,0,0, &  !v_sens
                                                               1,2,7,5,0,0,0, &  !qv_sens
                                                               5,0,0,0,0,0,0  &  ! time
#ifdef HOMME_AMDIS_PROJECT
                                                              ,1,2,5,0,0,0,0, &  ! dp
                                                               1,2,5,0,0,0,0, &  ! Q5
                                                               1,2,5,0,0,0,0, &  ! Q6
                                                               1,2,5,0,0,0,0, &  ! FM_x
                                                               1,2,5,0,0,0,0, &  ! FM_y
                                                               1,2,5,0,0,0,0, &  ! FM_z
                                                               1,2,5,0,0,0,0, &  ! FT
                                                               1,2,5,0,0,0,0, &  ! FQ1
                                                               1,2,5,0,0,0,0, &  ! FQ2
                                                               1,2,5,0,0,0,0  &  ! FQ3
#endif
                                                               /),&
                                                               shape=(/maxdims,varcnt/))

  integer, parameter :: vartype(varcnt)=(/nf_double, nf_double, nf_double,nf_double, nf_double,nf_double,nf_double,& !ps:cv_lon
                                          nf_int,    nf_double,nf_double,nf_double,nf_double,& !corners:T
                                          nf_double, nf_double,nf_double,nf_double,nf_double,nf_double,& !Th:w
                                          nf_double, nf_double, nf_double,nf_double,&
                                          nf_double, nf_double,nf_double,nf_double,nf_double,& !Q:geo
                                          nf_double, nf_double,nf_double,nf_double,nf_double,nf_double,& !omega:ilev
                                          nf_double, nf_double,nf_double,nf_double,& ! hybrid coords
                                          nf_double,nf_double,nf_double,nf_double,& !sentitivities
                                          nf_double  &  !time
#ifdef HOMME_AMDIS_PROJECT
                                         ,nf_double,nf_double,nf_double,& !dp, Q5, Q6
                                          nf_double,nf_double,nf_double,& !FM_x, FM_y, FM_z
                                          nf_double,& !FT
                                          nf_double,nf_double,nf_double  & !FQ1, FQ2, FQ3
#endif
                                          /)
  logical, parameter :: varrequired(varcnt)=(/.false.,.false.,.false.,.false.,.false.,.false.,.false.,&
                                              .false.,.false.,.false.,.false.,.false.,&
                                              .false.,.false.,.false.,.false.,.false.,.false.,&
                                              .false.,.false.,.false.,.false.,&
                                              .false.,.false.,.false.,.false.,.false.,&
                                              .false.,.false.,.true. ,.true. ,&
                                              .true. ,.true. ,&   ! lev,ilev
                                              .true. ,.true. ,&   ! hy arrays
                                              .true. ,.true. ,&   ! hy arrays
                                              .false.,.false.,.false.,.false.,& ! sensitivities
                                              .true.  &            ! time
#ifdef HOMME_AMDIS_PROJECT
                                             ,.false.,.false.,.false.,& ! dp, Q5, Q6
                                              .false.,.false.,.false.,& ! FM_x, FM_y, FM_z
                                              .false.,& ! FT
                                              .false.,.false.,.false.  & ! FQ1, FQ2, FQ3
#endif
                                              /)
  character*(*),parameter :: dimnames(maxdims)=(/'ncol        ',&
                                                 'lev         ',&
                                                 'ilev        ',&
                                                 'nelem       ',&
                                                 'time        ',&
                                                 'nsubelements',&
                                                 'deriv       '/)  
#else
  integer, parameter :: varcnt = 12
  integer, parameter :: maxdims=4
  character*(*),parameter::dimnames(maxdims)=(/'ncol ','nlev ','nelem','time '/)  
  integer, parameter :: vardims(maxdims,varcnt) =  reshape( (/ 1,4,0,0,  &  !ps
                                                               1,2,4,0,  &  !geop
                                                               1,2,4,0,  &  !u
                                                               1,2,4,0,  &  !v
                                                               1,0,0,0,  &  !lon (x for planar)
                                                               1,0,0,0,  &  !lat (y for planar)
                                                               4,0,0,0,  &  ! time
                                                               1,2,4,0,  &  ! zeta
                                                               1,2,4,0,  &  ! div
                                                               1,2,4,0,  &  ! eta = absolute vorticity = zeta + coriolis
                                                               1,2,4,0,  &  ! pv
                                                               1,0,0,0/),&  ! area
                                                               shape=(/maxdims,varcnt/))
  character*(*),parameter::varnames(varcnt)=(/'ps       ','geop     ','u        ','v        ',&
                                              'lon      ','lat      ','time     ',&
                                              'zeta     ','div      ','eta      ','pv       ','area     '/)
  integer, parameter :: vartype(varcnt)=(/nf_double,nf_double,nf_double,nf_double,nf_double,&
         nf_double,nf_double,nf_double,nf_double,nf_double,nf_double,nf_double/)
  logical, parameter :: varrequired(varcnt)=(/.false.,.false.,.false.,.false.,&
                                              .true.,.true.,.true.,&
                                              .false.,.false.,.false.,.false.,.true./)
#endif

  ! end of analysis_nl namelist variables

#endif

contains

!
! This gets the default var list for namelist_mod
!
  subroutine setvarnames(nlvarnames)
    character*(*), intent(out) :: nlvarnames(:)
#ifndef HOMME_WITHOUT_PIOLIBRARY
    integer :: lvarcnt
    if (varcnt > max_output_variables) then
       print *,__FILE__,__LINE__,"varcnt > max_output_varnames"
       stop
    endif
    lvarcnt=varcnt
    nlvarnames(1:varcnt) = varnames
    !print *,__FILE__,__LINE__,varcnt, size(nlvarnames),varnames
#endif
  end subroutine setvarnames

!
! This function returns the next step number in which an output (either restart or movie) 
! needs to be written.
!
  integer function nextoutputstep(tl)
    use time_mod, only : Timelevel_t, nendstep  
    use control_mod, only : restartfreq

    type(timelevel_t), intent(in) :: tl
    integer :: ios, nstep(max_output_streams)

    nstep(:) = nEndStep
    do ios=1,max_output_streams
       if((output_frequency(ios) .gt. 0)) then
!          dont check start_time, since this will skip desired output when
!                  nstep <   start_time <=  nextoutputstep
!          if ((output_start_time(ios) .le. tl%nstep) .and. &
           if ((output_end_time(ios) .ge. tl%nstep)) then
             nstep(ios)=tl%nstep + output_frequency(ios) - &
                  MODULO(tl%nstep,output_frequency(ios))
          end if
       end if
    end do
    nextoutputstep=minval(nstep)
    if(restartfreq>0) then
       nextoutputstep=min(nextoutputstep,tl%nstep+restartfreq-MODULO(tl%nstep,restartfreq))    
    end if
 end function nextoutputstep

end module common_movie_mod
