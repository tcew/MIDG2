#include "mpi.h"
#include "fem.h"

int main(int argc, char **argv){

  Mesh *mesh;
  int procid, nprocs, maxNv;
  int k,n, sk=0;
  double minEz, maxEz, gminEz, gmaxEz;

  /* initialize MPI */
  MPI_Init(&argc, &argv);

  /* assign gpu */
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  printf("procid=%d , nprocs=%d\n", procid, nprocs);

#if 0
  /* nicely stop MPI */
    MPI_Finalize();

  /* end game */
  exit(0);
#endif

  /* (parallel) read part of fem mesh from file */
  mesh = ReadMesh3d(argv[1]);

  /* perform load balancing */
  LoadBalance3d(mesh);

  /* find element-element connectivity */
  FacePair3d(mesh, &maxNv);

  /* perform start up */
  StartUp3d(mesh);

  /* field storage (double) */
  double *Hx = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Hy = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Hz = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Ex = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Ey = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Ez = (double*) calloc(mesh->K*p_Np, sizeof(double));

  /* initial conditions */
  for(k=0;k<mesh->K;++k){
    for(n=0;n<p_Np;++n) {
      Hx[sk] = 0;
      Hy[sk] = 0;
      Hz[sk] = 0;
      Ex[sk] = 0;
      Ey[sk] = 0;
      Ez[sk] = cos(M_PI*mesh->x[k][n])*cos(M_PI*mesh->y[k][n])*cos(M_PI*mesh->z[k][n]);
      ++sk;
    }
  }

  double dt, gdt;

  /* initialize OCCA info */
  double InitOCCA3d(Mesh *mesh, int Nfields);
  dt = InitOCCA3d(mesh, p_Nfields);

  /* load data onto GPU */
  gpu_set_data3d(mesh->K, Hx, Hy, Hz, Ex, Ey, Ez);

  MPI_Allreduce(&dt, &gdt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  dt = .5*gdt/((p_N+1)*(p_N+1));

  //  if(mesh->procid==0)
    printf("dt = %f\n", dt);

  double FinalTime = .5;

  /* solve */
  MaxwellsRun3d(mesh, FinalTime, dt); 

  /* unload data from GPU */
  void gpu_get_data3d(int K,
		      double *d_Hx, double *d_Hy, double *d_Hz,
		      double *d_Ex, double *d_Ey, double *d_Ez);
  gpu_get_data3d(mesh->K, Hx, Hy, Hz, Ex, Ey, Ez);

  /* find maximum & minimum values for Ez */
  minEz=Ez[0], maxEz=Ez[0];

  for(k=0;k<mesh->K;++k) {
    for(n=0;n<p_Np;++n){
      int id = n + p_Np*k;
      minEz = (minEz>Ez[id])?Ez[id]:minEz;
      maxEz = (maxEz<Ez[id])?Ez[id]:maxEz;
    }
  }

  MPI_Reduce(&minEz, &gminEz, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&maxEz, &gmaxEz, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if(procid==0)
    printf("t=%f Ez in [ %g, %g ] \n", FinalTime, gminEz, gmaxEz );

  /* nicely stop MPI */
  MPI_Finalize();

  /* end game */
  exit(0);
}
