#include <stdio.h>
#include <mpi.h>
#include "fem.h"

#include <occa.hpp>

occa::device device;

occa::memory c_LIFT;
occa::memory c_DrDsDt;
occa::memory c_surfinfo;
occa::memory c_mapinfo;
occa::memory c_vgeo;
occa::memory c_Q; 
occa::memory c_partQ; 
occa::memory c_rhsQ; 
occa::memory c_resQ; 
occa::memory c_tmp;
occa::memory c_parmapOUT;

occa::kernel rkKernel;
occa::kernel volumeKernel;
occa::kernel surfaceKernel;
occa::kernel partialGetKernel;

template <typename T>
void diagnose_array(const char *message, occa::memory &c_a, int N){

  device.finish();
  
  T *h_a = (T*) calloc(N, sizeof(T));
  
  c_a.copyTo(h_a, N*sizeof(T));

  double suma = 0;
  for(int n=0;n<N;++n){
    suma += h_a[n];
  }

  printf("%s: sum = %17.15g\n", message, suma);
  
  free(h_a);
}




double InitOCCA3d(Mesh *mesh, int Nfields){
  
  int plat = 1, dev = 0;
  device.setup("OpenCL", plat, dev);

  /* Q  */
  int sz = mesh->K*(BSIZE)*p_Nfields*sizeof(float); 

  float *f_Q = (float*) calloc(mesh->K*BSIZE*p_Nfields, sizeof(float));

  c_Q    = device.malloc(sz, f_Q);
  c_rhsQ = device.malloc(sz, f_Q);
  c_resQ = device.malloc(sz, f_Q);

  printf("sz1= %d\n", sz);
  
  sz = mesh->parNtotalout*sizeof(float);
  c_tmp  = device.malloc(sz, f_Q); // should not use f_Q
  c_partQ = device.malloc(sz, f_Q);

  printf("sz2= %d\n", sz);

  /*  LIFT  */
  sz = p_Np*(p_Nfp)*p_Nfaces*sizeof(float);

  float *f_LIFT = (float*) malloc(sz);
  int skL = 0;
  for(int m=0;m<p_Nfp;++m){
    for(int n=0;n<p_Np;++n){
      for(int f=0;f<p_Nfaces;++f){
	f_LIFT[skL++] = mesh->LIFT[0][p_Nfp*p_Nfaces*n+(f+p_Nfaces*m)];
      }
    }
  }

  c_LIFT = device.malloc(sz, f_LIFT);
   
  /* DrDsDt */
  sz = BSIZE*BSIZE*4*sizeof(float);

  float* h_DrDsDt = (float*) calloc(BSIZE*BSIZE*4, sizeof(float));
  int sk = 0;
  /* note transposed arrays to avoid "bank conflicts" */
  for(int n=0;n<p_Np;++n){
    for(int m=0;m<p_Np;++m){
      h_DrDsDt[4*(m+n*BSIZE)+0] = mesh->Dr[0][n+m*p_Np];
      h_DrDsDt[4*(m+n*BSIZE)+1] = mesh->Ds[0][n+m*p_Np];
      h_DrDsDt[4*(m+n*BSIZE)+2] = mesh->Dt[0][n+m*p_Np];
    }
  }
   
  c_DrDsDt = device.malloc(sz, h_DrDsDt);
   
  free(h_DrDsDt);

  /* vgeo */
  double drdx, dsdx, dtdx;
  double drdy, dsdy, dtdy;
  double drdz, dsdz, dtdz, J;
  float *vgeo = (float*) calloc(12*mesh->K, sizeof(float));

  for(int k=0;k<mesh->K;++k){
    GeometricFactors3d(mesh, k, 
		       &drdx, &dsdx, &dtdx,
		       &drdy, &dsdy, &dtdy,
		       &drdz, &dsdz, &dtdz, &J);

    vgeo[k*12+0] = drdx; vgeo[k*12+1] = drdy; vgeo[k*12+2] = drdz;
    vgeo[k*12+4] = dsdx; vgeo[k*12+5] = dsdy; vgeo[k*12+6] = dsdz;
    vgeo[k*12+8] = dtdx; vgeo[k*12+9] = dtdy; vgeo[k*12+10] = dtdz;

  }

  sz = mesh->K*12*sizeof(float);
  c_vgeo = device.malloc(sz, vgeo);
   
  /* surfinfo (vmapM, vmapP, Fscale, Bscale, nx, ny, nz, 0) */
  int sz5 = mesh->K*p_Nfp*p_Nfaces*5*sizeof(float); 
  float* h_surfinfo = (float*) malloc(sz5); 

  int sz2 = mesh->K*p_Nfp*p_Nfaces*2*sizeof(int); 
  int* h_mapinfo = (int*) malloc(sz2); 
   
  /* local-local info */
  sk = 0;
  int skP = -1;
  double *nxk = BuildVector(mesh->Nfaces);
  double *nyk = BuildVector(mesh->Nfaces);
  double *nzk = BuildVector(mesh->Nfaces);
  double *sJk = BuildVector(mesh->Nfaces);

  double dt = 1e6;

  for(int k=0;k<mesh->K;++k){

    GeometricFactors3d(mesh, k, 
		       &drdx, &dsdx, &dtdx,
		       &drdy, &dsdy, &dtdy,
		       &drdz, &dsdz, &dtdz, &J);

    Normals3d(mesh, k, nxk, nyk, nzk, sJk);
     
    for(int f=0;f<mesh->Nfaces;++f){

      dt = min(dt, J/sJk[f]);
       
      for(int m=0;m<p_Nfp;++m){
	int n = m + f*p_Nfp + p_Nfp*p_Nfaces*k;
	int idM = mesh->vmapM[n];
	int idP = mesh->vmapP[n];
	int  nM = idM%p_Np; 
	int  nP = idP%p_Np; 
	int  kM = (idM-nM)/p_Np;
	int  kP = (idP-nP)/p_Np;
	idM = nM + Nfields*BSIZE*kM;
	idP = nP + Nfields*BSIZE*kP;
	 
	/* stub resolve some other way */
	if(mesh->vmapP[n]<0){
	  idP = mesh->vmapP[n]; /* -ve numbers */
	}
 
	sk = 2*p_Nfp*p_Nfaces*k+m+f*p_Nfp;
	h_mapinfo[sk + 0*p_Nfp*p_Nfaces] = idM;
	h_mapinfo[sk + 1*p_Nfp*p_Nfaces] = idP;

	sk = 5*p_Nfp*p_Nfaces*k+m+f*p_Nfp;
	h_surfinfo[sk + 0*p_Nfp*p_Nfaces] = sJk[f]/(2.*J);
	h_surfinfo[sk + 1*p_Nfp*p_Nfaces] = (idM==idP)?-1.:1.;
	h_surfinfo[sk + 2*p_Nfp*p_Nfaces] = nxk[f];
	h_surfinfo[sk + 3*p_Nfp*p_Nfaces] = nyk[f];
	h_surfinfo[sk + 4*p_Nfp*p_Nfaces] = nzk[f];
      }
    }
  }
   
  c_mapinfo = device.malloc(sz2, h_mapinfo);
  c_surfinfo = device.malloc(sz5, h_surfinfo);

  free(h_mapinfo);
  free(h_surfinfo);

  printf("mesh->parNtotalout=%d\n", mesh->parNtotalout);
  sz = mesh->parNtotalout*sizeof(int);
  c_parmapOUT = device.malloc(sz, mesh->parmapOUT);

  /* now build kernels */
  occa::kernelInfo dgInfo;
   
  dgInfo.addDefine("p_Np",      p_Np);
  dgInfo.addDefine("p_Nfp",     p_Nfp);
  dgInfo.addDefine("p_Nfaces",  p_Nfaces);
  dgInfo.addDefine("p_Nfields", p_Nfields);
  dgInfo.addDefine("BSIZE",     BSIZE);

  int Nt, Nb;
  {
    volumeKernel = device.buildKernelFromSource("src/MaxwellsVolumeKernel3D.occa", 
						"MaxwellsVolumeKernel3D",
						dgInfo);
    Nt = p_Np;
    Nb = mesh->K;
    volumeKernel.setWorkingDims(1, Nt, Nb);
  }
  
  {
    surfaceKernel = device.buildKernelFromSource("src/MaxwellsSurfaceKernel3D.occa", 
						 "MaxwellsSurfaceKernel3D",
						 dgInfo);
    
    Nt = max(p_Np, p_Nfaces*p_Nfp);
    Nb = mesh->K;
    surfaceKernel.setWorkingDims(1, Nt, Nb);
  }

  {
    rkKernel = device.buildKernelFromSource("src/MaxwellsRKKernel3D.occa", 
					    "MaxwellsRKKernel3D",
					    dgInfo);
    
    int Ntotal = p_Nfields*BSIZE*mesh->K;
    Nt = 256;
    Nb = (Ntotal+Nt-1)/Nt;
    rkKernel.setWorkingDims(1, Nt, Nb);
  }

  {
    partialGetKernel = device.buildKernelFromSource("src/MaxwellsPartialGetKernel3D.occa",
						    "MaxwellsPartialGetKernel3D",
						    dgInfo);

    int Ntotal = mesh->parNtotalout;
    Nt = 256;
    Nb = (Ntotal+Nt-1)/Nt;
    partialGetKernel.setWorkingDims(1, Nt, Nb);
  }

#if 0
  diagnose_array<float>("c_DrDsDt", c_DrDsDt, 4*BSIZE*BSIZE);
  diagnose_array<float>("c_LIFT", c_LIFT, p_Nfaces*p_Nfp*p_Np);
  diagnose_array<float>("c_vgeo", c_vgeo, mesh->K*12);
  diagnose_array<float>("c_surfinfo", c_surfinfo, p_Nfaces*p_Nfp*7*mesh->K);
  diagnose_array<int>  ("c_parmapOUT", c_parmapOUT, mesh->parNtotalout);
#endif
  
  return dt;
}


/* assumes data resides on device */
void MaxwellsKernel3d(Mesh *mesh, float frka, float frkb, float fdt){

  /* grab data from device and initiate sends */
  void MaxwellsMPISend3d(Mesh *mesh);
  MaxwellsMPISend3d(mesh);

  /* evaluate volume derivatives */
  volumeKernel(mesh->K, c_vgeo, c_DrDsDt, c_Q, c_rhsQ);

#if 0
  diagnose_array<float>("c_Q", c_Q, p_Nfields*BSIZE*mesh->K);
  diagnose_array<float>("c_resQ", c_resQ, p_Nfields*BSIZE*mesh->K);
  diagnose_array<float>("c_rhsQ", c_rhsQ, p_Nfields*BSIZE*mesh->K);
#endif

  /* finalize sends and recvs, and transfer to device */
  void MaxwellsMPIRecv3d(Mesh *mesh);
  MaxwellsMPIRecv3d(mesh);

  /* evaluate surface contributions */
  surfaceKernel(mesh->K, c_mapinfo, c_surfinfo, c_LIFT, c_Q, c_partQ, c_rhsQ);

  /* update RK Step */
  int Ntotal = p_Nfields*BSIZE*mesh->K;
  rkKernel(Ntotal, c_resQ, c_rhsQ, c_Q, frka, frkb, fdt);

  device.finish();
}




void gpu_set_data3d(int K,
		    double *d_Hx, double *d_Hy, double *d_Hz,
		    double *d_Ex, double *d_Ey, double *d_Ez){


  float *f_Q = (float*) calloc(K*p_Nfields*BSIZE,sizeof(float));
  
  /* also load into usual data matrices */
  
  for(int k=0;k<K;++k){
    int gk = k;
    for(int n=0;n<p_Np;++n)
      f_Q[n        +k*BSIZE*p_Nfields] = d_Hx[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n  +BSIZE+k*BSIZE*p_Nfields] = d_Hy[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n+2*BSIZE+k*BSIZE*p_Nfields] = d_Hz[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n+3*BSIZE+k*BSIZE*p_Nfields] = d_Ex[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n+4*BSIZE+k*BSIZE*p_Nfields] = d_Ey[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n+5*BSIZE+k*BSIZE*p_Nfields] = d_Ez[n+gk*p_Np];
  }

  c_Q.copyFrom(f_Q);

  free(f_Q);
}
  
void gpu_get_data3d(int K,
		    double *d_Hx, double *d_Hy, double *d_Hz,
		    double *d_Ex, double *d_Ey, double *d_Ez){

  float *f_Q = (float*) calloc(K*p_Nfields*BSIZE,sizeof(float));

  c_Q.copyTo(f_Q, K*p_Nfields*BSIZE*sizeof(float));

  /* also load into usual data matrices */
  
  for(int k=0;k<K;++k){
    int gk = k;
    for(int n=0;n<p_Np;++n)
      d_Hx[n+gk*p_Np] = f_Q[n        +k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n) 
      d_Hy[n+gk*p_Np] = f_Q[n  +BSIZE+k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n)
      d_Hz[n+gk*p_Np] = f_Q[n+2*BSIZE+k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n)
      d_Ex[n+gk*p_Np] = f_Q[n+3*BSIZE+k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n) 
      d_Ey[n+gk*p_Np] = f_Q[n+4*BSIZE+k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n)
      d_Ez[n+gk*p_Np] = f_Q[n+5*BSIZE+k*BSIZE*p_Nfields];

  }

  free(f_Q);
}


void get_partial_gpu_data3d(int Ntotal, float *h_partQ){

  device.finish();

  partialGetKernel (Ntotal, c_Q, c_parmapOUT, c_tmp);

  c_tmp.copyTo(h_partQ); // , Ntotal*sizeof(float));

}


static MPI_Request *mpi_out_requests = NULL;
static MPI_Request *mpi_in_requests  = NULL;

static int Nmess = 0;

void MaxwellsMPISend3d(Mesh *mesh){

  int p;

  int procid = mesh->procid;
  int nprocs = mesh->nprocs;

  MPI_Status status;

  if(mpi_out_requests==NULL){
    mpi_out_requests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
    mpi_in_requests  = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  }

  if(mesh->parNtotalout){

#if 0
    for(int n=0;n<mesh->parNtotalout;++n){
      printf("f_outQ[%d] =%g\n", n, mesh->f_outQ[n]);
    }
#endif


    void get_partial_gpu_data3d(int Ntotal, float *h_partQ);
    get_partial_gpu_data3d(mesh->parNtotalout, mesh->f_outQ);

  }

  /* non-blocked send/recv partition surface data */
  Nmess = 0;

  /* now send piece to each proc */
  int sk = 0;
  for(p=0;p<nprocs;++p){

    if(p!=procid){
      int Nout = mesh->Npar[p]*p_Nfields*p_Nfp;
      if(Nout){
	/* symmetric communications (different ordering) */
	MPI_Isend(mesh->f_outQ+sk, Nout, MPI_FLOAT, p, 6666+p,      MPI_COMM_WORLD, mpi_out_requests +Nmess);
	MPI_Irecv(mesh->f_inQ+sk,  Nout, MPI_FLOAT, p, 6666+procid, MPI_COMM_WORLD,  mpi_in_requests +Nmess);
	sk+=Nout;
	++Nmess;
      }
    }
  }

}


void MaxwellsMPIRecv3d(Mesh *mesh){
  int p, n;
  int nprocs = mesh->nprocs;

  MPI_Status *instatus  = (MPI_Status*) calloc(nprocs, sizeof(MPI_Status));
  MPI_Status *outstatus = (MPI_Status*) calloc(nprocs, sizeof(MPI_Status));

  MPI_Waitall(Nmess, mpi_in_requests, instatus);

  if(mesh->parNtotalout)
    c_partQ.copyFrom(mesh->f_inQ); // , mesh->parNtotalout*sizeof(float));

  MPI_Waitall(Nmess, mpi_out_requests, outstatus);

  free(outstatus);
  free(instatus);

}

