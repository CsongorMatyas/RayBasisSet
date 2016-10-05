      subroutine NFC_matrix (FC_matrix, FCdim)
!*****************************************************************************************************************
!     Date last modified: February 18, 2000                                                         Version 2.0  *
!     Author: C. C. Pye and R. A. Poirier                                                                        *
!     Description: Calculate the Hessian numerically.                                                            *
!*****************************************************************************************************************
! Modules:
      USE program_files
      USE program_constants
      USE OPT_defaults
      USE matrix_print
      USE OPT_objects
      USE mod_type_hessian
      USE QM_defaults

      implicit none
!
! Input scalar:
      integer :: FCdim
      double precision :: FC_matrix(FCdim,FCdim)
!
! Local scalars:
      integer I,J,IPARAM,status
      double precision SQRFSQ
      double precision :: SCFACC_save
      logical LRESET
!
! Local work arrays:
      double precision, dimension(:,:), allocatable :: GRD_forward
      double precision, dimension(:,:), allocatable :: GRD_backward
      double precision, dimension(:), allocatable :: PARSET0
      double precision, dimension(:), allocatable :: PARGRD0
!
! Begin:
      call PRG_manager ('enter', 'NFC_matrix', 'UTILITY')
!
! Set higher SCF accuracy for Force constant evaluation
      SCFACC_save=SCFACC
!      SCFACC=1.0D-07

      NFEVALH=0
      if(Noptpr.ne.FCdim)then
        write(UNIout,'(a)')'NFC_matrix> Force constant and parameter set dimension do not match'
        call PRG_stop ('NFC_matrix> Force constant and parameter set dimension do not match')
      end if

      call GET_object ('OPT', 'HESSIAN', 'STEP_SIZE')
!
      IF(LFDIFF.or.LBDIFF)then ! Check if any numerical differentiation required

      NFEVALH=NFEVALH+1 ! Count the pivot point
!
      call GET_object ('OPT', 'FUNCTION', OPT_function%name) ! get a function value
      call GET_object ('OPT', 'GRADIENTS', OPT_parameters%name)
!
      allocate (GRD_forward(1:Noptpr,1:Noptpr), GRD_backward(1:Noptpr,1:Noptpr), STAT=status)
      allocate (PARSET0(1:Noptpr), PARGRD0(1:Noptpr), STAT=status) ! NOTE: may already be allocated (ignore)

      PARSET0(1:Noptpr)=PARSET(1:Noptpr)
      PARGRD0(1:Noptpr)=PARGRD(1:Noptpr)

! Print results:
        IF(LprintHeval)then
        SQRFSQ=ZERO
        do I=1,Noptpr
          SQRFSQ=SQRFSQ+PARGRD0(I)*PARGRD0(I)
        end DO
        SQRFSQ=DSQRT(SQRFSQ/DBLE(Noptpr))

!       if(rank.eq.0)then
        J=MIN0(Noptpr,10)
        write(UNIout,'(a,F15.7,a,1PE16.8/a,10(1PE13.5))')' Pivot point: Energy = ',OPT_function%value, &
                     ',  GRADIENT LENGTH = ',SQRFSQ,' X:',(PARSET0(I),I=1,J)
        IF(Noptpr.GT.10)write(UNIout,'(3X,1PE13.5,9E13.5)')(PARSET0(I),I=11,Noptpr)
        J=MIN0(Noptpr,10)
        write(UNIout,'(a,10(1PE13.5))')' F:',(PARGRD0(I),I=1,J)
        IF(Noptpr.GT.10)write(UNIout,'(3X,1PE13.5,9E13.5)')(PARGRD0(I),I=11,Noptpr)
        end if
!       end if

! Loop over rows of Hessian for Forward differences
      if(LFDIFF)then
      do IPARAM=1,Noptpr
        LRESET=.false.

! New parameter set
      IF(HESTYP(IPARAM).EQ.'FDIFF'.or.HESTYP(IPARAM).EQ.'CDIFF')then
        NFEVALH=NFEVALH+1
        call GRD_FSET (PARSET, PARSET0, DSTEP, Noptpr, IPARAM)
        LRESET=.TRUE.
      end IF

! Reset all objects if new gradient required
        IF(LRESET)then
          call OPT_TO_COOR
          call GET_object ('OPT', 'FUNCTION', OPT_function%name) ! get a new function value
          call GET_object ('OPT', 'GRADIENTS', OPT_parameters%name) ! get new gradients
        end IF

! Save parameter gradients in GRD_forward and set a new parameter value
        call SAVE_GRD (PARSET, PARGRD, GRD_forward, Noptpr, IPARAM, LprintHeval, OPT_function%value, LRESET)

      end do ! IPARAM=,Noptpr
      end IF ! IF(LFDIFF)

! Loop over rows of Hessian for Backward differences
      IF(LBDIFF)then
      do IPARAM=1,Noptpr
        LRESET=.false.
!
! New parameter set
      IF(HESTYP(IPARAM).EQ.'BDIFF'.or.HESTYP(IPARAM).EQ.'CDIFF')then
        NFEVALH=NFEVALH+1
        call GRD_BSET (PARSET, PARSET0, DSTEP, Noptpr, IPARAM)
        LRESET=.TRUE.
      end IF

! Reset all objects if new gradient required
        IF(LRESET) then
          call OPT_TO_COOR
          call GET_object ('OPT', 'FUNCTION', OPT_function%name) ! get a new function value
          call GET_object ('OPT', 'GRADIENTS', OPT_parameters%name) ! get new gradients
        end IF

! save parameter gradients in GRD_backward and set a new parameter value
          call SAVE_GRD (PARSET, PARGRD, GRD_backward, Noptpr, IPARAM, LprintHeval, OPT_function%value, LRESET)
!
      end do ! IPARAM=1,Noptpr
      end IF ! IF(LBDIFF)
!
      write(UNIout,'(a)')'Numerical Hessian evaluation complete'
      write(UNIout,'(a,i10)')'Total number of function evaluations: ',NFEVALH
!
      call BLD_Hessian_by_diff (HESTYP, PARGRD0, DSTEP, FC_matrix, GRD_forward, &
                                GRD_backward, Noptpr)

! Reset parameters to the pivot point
      PARSET(1:Noptpr)=PARSET0(1:Noptpr)
      PARGRD(1:Noptpr)=PARGRD0(1:Noptpr)
      call OPT_TO_COOR
!
      if(Lcubic)then ! Calculated the semi-third derivate
        call SEMID_THDER
      end if
!
! Do not deallocate GRD_forward and GRD_backward
      deallocate (PARSET0, PARGRD0)
      deallocate (GRD_forward, GRD_backward)
!
      call FC_SYMMETRIZE (FC_matrix, Noptpr, NnegEig)
      else ! use default Hessian
!       write(UNIout,'(/a)')'Hessian (default) in Hartree/bohr**2'
!       call PRT_hessian (FC_matrix, Noptpr)
      end IF ! LFDIFF.or.LBDIFF

! Reset SCF accuracy
      SCFACC=SCFACC_save
!
! End of routine NFC_matrix
      call PRG_manager ('exit', 'NFC_matrix', 'UTILITY')
      return
      contains
      subroutine SEMID_THDER
!***********************************************************************
!     Date last modified: February 15, 1995               Version 1.0  *
!     Author: Cory C. Pye                                              *
!     Description: Calculates semidiagonal third derivative            *
!                  approximation from forward and backward gradient    *
!                  sets.                                               *
!     T_kkl = (g_l(x_0+l_k e_k)+g_l(x_0-l_k e_k)-2g_l(x_0))/(l_k)^2.   *
!     Ref: C. C. Pye, Book II p. 7                                     *
!          P. Pulay, G. Fogarasi, F. Pang, J. E. Boggs, J. Am. Chem.   *
!          Soc., 101 (1979) p2550 (in particular, p 2555, eq 6)        *
!***********************************************************************

      implicit none
!
! Local scalars:
      integer I,J
!
! Begin:
      call PRG_manager ('enter', 'SEMID_THDER', 'OPT:THIRD_DERIVATIVES%SEMI_DIAGONAL')
!
      if(.not.allocated(STHIRD))then
        allocate (STHIRD(1:Noptpr,1:Noptpr), STAT=status)
      end if
!
! Whcih pargrd should be used here?????????????
      do I=1, Noptpr
        IF(HESTYP(I)(1:5).EQ.'CDIFF')then  ! Use central difference formula
          do J=1, Noptpr
            STHIRD(I,J)=(GRD_forward(I,J)+GRD_backward(I,J)-TWO*PARGRD(J))/(DSTEP(I)*DSTEP(I))
          end do ! J=1
        else                               ! Central difference formula not used
          do J=1, Noptpr
            STHIRD(I,J)=ZERO
          end DO
        end IF ! HESTYP(I).EQ.'CDIFF'
      end do ! I=1
!
! End of routine SEMID_THDER
      call PRG_manager ('exit', 'SEMID_THDER', 'OPT:THIRD_DERIVATIVES%SEMI_DIAGONAL')
      return
      end subroutine SEMID_THDER
      end subroutine NFC_matrix
      subroutine BLD_Hessian_by_diff (HESTYP, & ! Hessian Type by parameter
                         PARGRD0, & ! Pivot Parameter Gradients
                         DSTEP , & ! Step vector from pivot
                         Hessian_matrix, & ! Hessian Guess
                         GRD_forward, & ! Forward difference gradients
                         GRD_backward, & ! Bacward difference gradients
                         Noptpr)! Number of optimizable parameters
!***********************************************************************
!     Date last modified: February 5, 1999                             *
!     Author: C. C. Pye and R. A. Poirier                              *
!     Description: Calculates Hessian approximation from forward and   *
!                  backward gradient sets.                             *
!***********************************************************************
! Modules:
      USE OPT_defaults
      USE program_constants
      USE matrix_print

      implicit none
!
! Input scalars:
      integer Noptpr
!
! Input arrays:
      double precision Hessian_matrix(Noptpr,Noptpr),DSTEP(Noptpr),PARGRD0(Noptpr)
      double precision GRD_forward(Noptpr,Noptpr),GRD_backward(Noptpr,Noptpr)
      character*8 HESTYP(Noptpr)
!
! Local scalars:
      integer I,J
      double precision TEMP
!
! Begin:
      call PRG_manager ('enter', 'BLD_Hessian_by_diff', 'UTILITY')
!
! Now compute the Hessian using finite differences
      do I=1,Noptpr
      DIFFERENCE: select case (HESTYP(I))
        case ('FDIFF')
          do J=1,Noptpr
            Hessian_matrix(I,J)=(GRD_forward(I,J)-PARGRD0(J))/DSTEP(I) ! J -> I in DSTEP RAP (May 5/98)
          end do ! J
        case ('BDIFF')
          do J=1,Noptpr
            Hessian_matrix(I,J)=(GRD_backward(I,J)-PARGRD0(J))/(-DSTEP(I)) ! same
          end do ! J
        case ('CDIFF')
          do J=1,Noptpr
            Hessian_matrix(I,J)=(GRD_forward(I,J)-GRD_backward(I,J))/(TWO*DSTEP(I)) ! same
          end do ! J
        case ('IDENT')
          Hessian_matrix(I,I)=ONE
      end select DIFFERENCE
      end do ! I=1
!
! If off-diagonal H_ij is not calculated, but H_ji is, copy H_ji->H_ij
      do I=1,Noptpr
        IF(HESTYP(I)(2:5).NE.'DIFF')then ! Not calculated
          do J=1,Noptpr
            IF(HESTYP(J)(2:5).EQ.'DIFF')then ! Not Calculated
              Hessian_matrix(I,J)=Hessian_matrix(J,I)
            end IF
          end DO
        end IF
      end DO
!
! End of routine BLD_Hessian_by_diff
      call PRG_manager ('exit', 'BLD_Hessian_by_diff', 'UTILITY')
      return
      end
      subroutine GRD_FSET (PARSET,   & ! Current parameter set
                           OLDPAR,   & ! The 'pivot' point
                           DSTEP,    & ! Size of perturbation
                           Noptpr,   & ! Number of optimizable parameters
                           IPARAM)  ! Index of gradient to be added
!***********************************************************************
!     Date last modified: December  3, 1993               Version 1.0  *
!     Author: Cory C. Pye                                              *
!     Description: Copies current gradients into gradient matrix and   *
!                  updates the current parameter set.                  *
!***********************************************************************
! Modules:
      USE OPT_defaults

      implicit none
!
! Input scalars:
      integer Noptpr,IPARAM
!
! Input arrays:
      double precision PARSET(Noptpr),OLDPAR(Noptpr),DSTEP(Noptpr)
!
! Begin:
      call PRG_manager ('enter', 'GRD_FSET', 'UTILITY')
!
      PARSET(1:Noptpr) = OLDPAR(1:Noptpr)
! Update parameter set
      PARSET(IPARAM)=PARSET(IPARAM)+DSTEP(IPARAM)
!
! End of routine GRD_FSET
      call PRG_manager ('exit', 'GRD_FSET', 'UTILITY')
      return
      end
      subroutine GRD_BSET (PARSET,   & ! Current parameter set
                         OLDPAR,   & ! The 'pivot' point
                         DSTEP,    & ! Size of perturbation
                         Noptpr,   & ! Number of optimizable parameters
                         IPARAM)  ! Index of gradient to be added
!***********************************************************************
!     Date last modified: December 10, 1993               Version 1.0  *
!     Author: Cory C. Pye                                              *
!     Description: Copies current gradients into gradient matrix and   *
!                  updates the current parameter set.                  *
!***********************************************************************
! Modules:
      USE OPT_defaults

      implicit none
!
! Input scalars:
      integer Noptpr,IPARAM
!
! Input arrays:
      double precision PARSET(Noptpr),OLDPAR(Noptpr),DSTEP(Noptpr)
!
! Begin:
      call PRG_manager ('enter', 'GRD_BSET', 'UTILITY')
!
      PARSET(1:Noptpr) = OLDPAR(1:Noptpr)

! Update parameter set
      PARSET(IPARAM)=PARSET(IPARAM)-DSTEP(IPARAM)
!
! End of routine GRD_BSET
      call PRG_manager ('exit', 'GRD_BSET', 'UTILITY')
      return
      end
      subroutine SAVE_GRD (PARSET,  & ! Current parameters
                           PARGRD,   & ! Current gradient
                           GRDSET,   & ! Set of all gradients ('guess')
                           Noptpr,   & ! Number of optimizable parameters
                           IPARAM,   & ! Index of gradient to be added
                           LPRINT,   & ! Print function evaluations?
                           OPT_function_value,     & ! Current total energy
                           LRESET)  ! Reset the next parameters?
!***********************************************************************
!     Date last modified: January 28, 1999                Version 1.0  *
!     Author: R. A. Poirier                                            *
!     Description: Copies current gradients into gradient matrix       *
!***********************************************************************
! Modules:
      USE program_files
      USE program_constants
      USE OPT_defaults

      implicit none
!
! Input scalars:
      integer Noptpr,IPARAM
      double precision OPT_function_value
      logical LPRINT
      logical LRESET
!
! Input arrays:
      double precision PARSET(Noptpr),PARGRD(Noptpr),GRDSET(Noptpr,Noptpr)
!
! Local scalars:
      integer I,J
      double precision SQRFSQ
!
! Begin:
      call PRG_manager ('enter', 'SAVE_GRD', 'UTILITY')
!
! Update gradient matrix
      IF(LRESET)then
        GRDSET(IPARAM,1:Noptpr)=PARGRD(1:Noptpr)
      else
        GRDSET(IPARAM,1:Noptpr)=ZERO
      end IF
!
! Print results:
        IF(LRESET.and.LPRINT)then
        SQRFSQ=ZERO
        do I=1,Noptpr
          SQRFSQ=SQRFSQ+PARGRD(I)*PARGRD(I)
        end DO
        SQRFSQ=DSQRT(SQRFSQ/DBLE(Noptpr))
        J=MIN0(Noptpr,10)
        write(UNIout,'(a,I5,a,F15.7,a,1PE16.8/a,10(1PE13.5))') &
             'At Parameter ',IPARAM,', Energy = ',OPT_function_value, &
             ',  GRADIENT LENGTH = ',SQRFSQ,' X:',(PARSET(I),I=1,J)
        IF(Noptpr.GT.10)write(UNIout,'(3X,1PE13.5,9E13.5)')(PARSET(I),I=11,Noptpr)
        J=MIN0(Noptpr,10)
        write(UNIout,'(a,10(1PE13.5))')' F:',(PARGRD(I),I=1,J)
        IF(Noptpr.GT.10)write(UNIout,'(3X,1PE13.5,9E13.5)')(PARGRD(I),I=11,Noptpr)
        end IF
!
! End of routine SAVE_GRD
      call PRG_manager ('exit', 'SAVE_GRD', 'UTILITY')
      return
      end
      subroutine FCSTEP
!***********************************************************************
!     Date last modified: October 29 , 1993               Version 1.0  *
!     Author: C.C. Pye and R.A. Poirier                                *
!     Description: Sets default step length of all parameters for      *
!                  finite-difference Hessian approximation to a value  *
!                  of DFLTSTEP                                         *
!***********************************************************************
! Modules:
      USE OPT_defaults
      USE OPT_objects
      USE mod_type_hessian

      implicit none
!
! Local scalars:
      integer IPARAM
!
! Begin:
      call PRG_manager ('enter', 'FCSTEP', 'UTILITY')
!
      do IPARAM=1,Noptpr
        DSTEP(IPARAM)=DFLTSTEP
      end DO
!
! End of routine FCSTEP
      call PRG_manager ('exit', 'FCSTEP', 'UTILITY')
      return
      end
      subroutine GRD_SET (GRD_forward, GRD_backward)
!***********************************************************************
!     Date last modified: December  3 , 1993              Version 1.0  *
!     Author: Cory C. Pye                                              *
!     Description: Get addresses for GRD_FSET.                           *
!     This is just a copy of Hessian_numerical (improve)                          *
!***********************************************************************
! Modules:
      USE program_files
      USE OPT_defaults
      USE OPT_objects
      USE mod_type_hessian

      implicit none
!
! Outpur arrays:
      double precision :: GRD_forward(Noptpr,Noptpr),GRD_backward(Noptpr,Noptpr)
!
! Local scalars:
      integer IPARAM,status
      logical LRESET
!
! Local work arrays:
      double precision, dimension(:), allocatable :: PARSET0
!
! Begin:
      call PRG_manager ('enter', 'GRD_SET', 'UTILITY')
!
! Local work array:
      allocate (PARSET0(1:Noptpr), STAT=status) ! NOTE: may already be allocate (ignore)

      PARSET0(1:Noptpr)=PARSET(1:Noptpr)
!
! Loop over rows of Hessian for Forward differences
      IF(LFDIFF)then
      do IPARAM=1,Noptpr
       LRESET=.false.
!
! New parameter set
      IF(HESTYP(IPARAM).EQ.'FDIFF'.or.HESTYP(IPARAM).EQ.'CDIFF')then
        NFEVALH=NFEVALH+1
        call GRD_FSET (PARSET, PARSET0, DSTEP, Noptpr, IPARAM)
        LRESET=.TRUE.
      end IF

! Reset all objects if new gradient required
        IF(LRESET) then
          call OPT_TO_COOR
          call GET_object ('OPT', 'FUNCTION', OPT_function%name) ! get a new function value
          call GET_object ('OPT', 'GRADIENTS', OPT_function%name) ! get new gradients
        end IF

! save parameter gradients in GRD_forward and set a new parameter value
          call SAVE_GRD (PARSET, PARGRD, GRD_forward, Noptpr, IPARAM, LprintHeval, OPT_function%value, LRESET)
!
      end do ! IPARAM=,Noptpr
      end IF ! IF(LFDIFF)

! Loop over rows of Hessian for Backward differences
      IF(LBDIFF)then
      do IPARAM=1,Noptpr
       LRESET=.false.
!
! New parameter set
      IF(HESTYP(IPARAM).EQ.'BDIFF'.or.HESTYP(IPARAM).EQ.'CDIFF')then
        NFEVALH=NFEVALH+1
        call GRD_BSET (PARSET, PARSET0, DSTEP, Noptpr, IPARAM)
        LRESET=.TRUE.
      end IF

! Reset all objects if new gradient required
        IF(LRESET) then
          call OPT_TO_COOR
          call GET_object ('OPT', 'FUNCTION', OPT_function%name) ! get a new function value
          call GET_object ('OPT', 'GRADIENTS', OPT_function%name) ! get new gradients
        end IF

! save parameter gradients in GRD_backward and set a new parameter value
          call SAVE_GRD (PARSET, PARGRD, GRD_backward, Noptpr, IPARAM, LprintHeval, OPT_function%value, LRESET)
!
      end do ! IPARAM=1,Noptpr
      end IF ! IF(LBDIFF)

      PARSET(1:Noptpr)=PARSET0(1:Noptpr)
      call OPT_TO_COOR
!
      write(UNIout,'(a)')'Numerical Hessian evaluation complete'
      write(UNIout,'(a,i10)')'Total number of function evaluations: ',NFEVALH
!
      deallocate (PARSET0)
!
! End of routine GRD_SET
      call PRG_manager ('exit', 'GRD_SET', 'UTILITY')
      return
      end
      subroutine FC_SYMMETRIZE (WJ,       & ! Force constant / Hessian
                                N,        & ! Dimension
                                NnegEig)    ! Number of negative eigvalues
!***********************************************************************
!     Date last modified: April 13, 1998                               *
!     Author: R. A. Poirier                                            *
!     Description:                                                     *
!     Symmetrize the force constant matrix and print it.               *
!     WFC is a copy of the Hessian.                                    *
!***********************************************************************
! Modules:
      USE program_files
      USE program_constants
      USE math_interface
      USE OPT_defaults
      USE OPT_objects
      USE matrix_print

      implicit none
!
! Input scalars:
      integer, intent(in) :: N
      integer, intent(out) :: NnegEig
!
! Input arrays:
      double precision WJ(N,N)
!
! Work arrays:
      double precision, dimension(:,:), allocatable :: WFC
      double precision, dimension(:), allocatable :: W
!
! Local scalars:
      integer I,J,JJ,K
      double precision T
!
! Local parameters:
      double precision TENM3
      parameter (TENM3=1.0D-5) ! Was change from -3 12/7/11
!
      allocate (W(N), WFC(N,N))
!
! Make WJ symmetrical.
      IF(LHESSYM)then
        do J=1,N
        do I=1,J
          T=PT5*(WJ(I,J)+WJ(J,I))
          WJ(I,J)=T
          WJ(J,I)=T
        end DO
        end DO
        write(UNIout,'(/a)')' Hessian is symmetrized'
      else
        write(UNIout,'(a)')' Hessian is unsymmetrized'
      end if

      if(OPT_parameters%name(1:8).ne.'SFACTORS')then
        call PRT_hessian (WJ, N)
      else
        call PRT_matrix (WJ, N, N)
      end if
!
! Get eigenvalues of Hessian
      WFC(1:N,1:N)=WJ(1:N,1:N)
      if(OPT_parameters%name(1:8).ne.'SFACTORS')call FC_convert (WFC, N)
      call MATRIX_diagonalize (WFC, WFC, W, -2, .true.)
      write(UNIout,'(/a)')'Eigenvalues of (un)symmetrized Hessian' ! (Mdyne/Angstrom)'
      do I=1,N
        write(UNIout,'(1X,I3,F15.6)')I,W(I)
      end do ! I
!
!     Find the order of the critical point.
      NnegEig=0
      JJ=0
      do I=1,N
        T=W(I)
        IF(DABS(T).LE.TENM3)then ! Zero eigenvalue.
          JJ=JJ+1
        else if(T.LT.ZERO)then ! Negative eigenvalue.
          NnegEig=NnegEig+1
        end IF
      end do ! I
!
      write(UNIout,'(/a,I3,a,I3,a)')' This critical point has order',NnegEig, &
                                     ' There are',JJ,' Degenerate eigenvalues'
!
      write(UNIout,'(/2a)')'Eigenvectors of symmetrized Hessian(as columns by ', &
                           'eigenvalue number, the rows are the optimized variable numbers)'
      call PRT_matrix (WFC, N, N)

      deallocate (W, WFC)
!
      return
      end
