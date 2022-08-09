subroutine read_cids(filename)
   use ISO_FORTRAN_ENV

   character(80)   :: filename
   integer(kind=int32)  :: ntraj, nparticles, cids
   !integer(kind=int32)  :: ntraj, nparticles, cids
   integer(kind=int32), allocatable, dimension(:,:) :: clusters

   integer :: itraj, jpart

   open(999,file=trim(filename),status='old',form='unformatted')
   read(999) ntraj
   read(999) nparticles
   print*, ntraj, nparticles
   allocate( clusters(int(ntraj),int(nparticles)) )
   read(999) clusters

   !print*, clusters(1,1), clusters(1,2), clusters(1,3)
   !print*, clusters(2,1), clusters(2,2), clusters(2,3)
end subroutine

program test_read
   ! Example
   call read_cids("cluster.cids")
end program


