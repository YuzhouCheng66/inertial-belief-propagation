// FP64 synchronous BP / I-BP; one graph is distributed across OpenMP threads.
// No BLAS, no task-level batching, no -ffast-math, fixed group reductions.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>
#include <time.h>
using V=std::vector<double>;
struct Data {
 int domain,L,N,G,seed,mode; double J; std::vector<int> nb,offset,order; V field,prior,diag,p,P,alpha,T;
 template<class A> static void read(std::ifstream& f,std::vector<A>& v,size_t n){v.resize(n);f.read((char*)v.data(),n*sizeof(A));if(!f)throw std::runtime_error("truncated fixture");}
 explicit Data(const std::string& name){
  std::ifstream f(name,std::ios::binary);int h[7];f.read((char*)h,sizeof h);if(h[0]!=734211)throw std::runtime_error("bad fixture");
  domain=h[1];L=h[2];N=h[3];G=h[4];seed=h[5];mode=h[6];f.read((char*)&J,8);
  read(f,nb,N*4);read(f,offset,G+1);read(f,order,N);read(f,field,N);read(f,prior,N);read(f,diag,N);
  read(f,p,N*4);read(f,P,N);read(f,alpha,N*4);if(domain==2)read(f,T,size_t(N)*256);
 }
};
inline double cpu_seconds(){timespec t;clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t);return t.tv_sec+1e-9*t.tv_nsec;}
inline void logvec(const double* m,double* x){
 double r=std::sqrt(m[0]*m[0]+m[1]*m[1]+m[2]*m[2]);
 double fac=r>1e-150?std::atanh(2*r)/r:2.;for(int a=0;a<3;a++)x[a]=fac*m[a];
}
inline void expvec(const double* x,double* m){
 double r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);double fac=r>1e-150?.5*std::tanh(r)/r:.5;
 for(int a=0;a<3;a++)m[a]=fac*x[a];
}
struct State {
 const Data& d; V a,b,c,ta,tb,tc,previous,beta,clock,delta,local1,local2,local3,x,resvec;
 double *cur,*nxt,*prop,*total,*totalNext,*totalProp; bool has=false;
 State(const Data& dd):d(dd){
  size_t M=size_t(d.N)*4*(d.domain==2?3:1);a.assign(M,0);b=a;c=a;previous=a;
  ta=d.field;tb=ta;tc=ta;cur=a.data();nxt=b.data();prop=c.data();total=ta.data();totalNext=tb.data();totalProp=tc.data();
  beta.assign(d.G,0);clock.assign(d.G,1);delta.assign(d.N,0);local1.assign(d.G,0);local2=local1;local3=local1;
  x.assign(d.N,0);resvec=x;if(d.domain==0)previous.assign(d.N,0);
 }
 void sweep(const double* in,const double* totals,double* out,double* outtotal){
  const int N=d.N;
  if(d.domain==2){
   #pragma omp for schedule(static)
   for(int i=0;i<N;i++){
    double inc[4][4];
    for(int p=0;p<4;p++){
     int j=d.nb[4*i+p],rev=(p+2)%4;const double* m=in+12*j+3*rev;
     inc[p][0]=.5;inc[p][1]=m[0];inc[p][2]=m[1];inc[p][3]=m[2];
    }
    double left[16],right[16],L[16]={},R[16]={};
    for(int a=0;a<4;a++)for(int b=0;b<4;b++){
     left[4*a+b]=inc[0][a]*inc[1][b];right[4*a+b]=inc[2][a]*inc[3][b];
    }
    const double* T=d.T.data()+size_t(i)*256;
    for(int a=0;a<16;a++){
     double s=0;
     for(int b=0;b<16;b++){double t=T[16*a+b];s+=t*right[b];L[b]+=t*left[a];}
     R[a]=s;
    }
    double v[4][4]={};
    for(int a=0;a<4;a++)for(int b=0;b<4;b++){
     v[0][a]+=R[4*a+b]*inc[1][b];v[1][b]+=R[4*a+b]*inc[0][a];
     v[2][a]+=L[4*a+b]*inc[3][b];v[3][b]+=L[4*a+b]*inc[2][a];
    }
    for(int p=0;p<4;p++){
     double f=.5/v[p][0];out[12*i+3*p]=f*v[p][1];out[12*i+3*p+1]=-f*v[p][2];out[12*i+3*p+2]=f*v[p][3];
    }
   }
  }else{
   const double coupling=std::tanh(d.J);
   #pragma omp for schedule(static)
   for(int i=0;i<N;i++){
    double s=d.field[i];
    for(int p=0;p<4;p++){
     int e=4*i+p,j=d.nb[e];double v=0;
     if(j>=0){double cavity=totals[j]-in[4*j+(p+2)%4];
      if(d.domain==0)v=d.alpha[e]*cavity;
      else v=std::atanh(std::max(-1.+1e-15,std::min(1.-1e-15,coupling*std::tanh(cavity))));
     }
     out[e]=v;s+=v;
    }
    outtotal[i]=s;
   }
  }
 }
 void residuals(){
  if(d.domain==0){
   #pragma omp for schedule(static)
   for(int i=0;i<d.N;i++)x[i]=total[i]/d.P[i];
  }
  #pragma omp for schedule(static)
  for(int g=0;g<d.G;g++){
   double err=0,mx=0,er2=0;
   for(int it=d.offset[g];it<d.offset[g+1];it++){
    int i=d.order[it];
    if(d.domain==2){
     for(int p=0;p<4;p++){
      const double* a=cur+12*i+3*p;const double* b=prop+12*i+3*p;
      double dr=std::hypot(a[0]-b[0],a[1]-b[1]);double dz=std::abs(a[2]-b[2]);
      err=std::max(err,std::max(dr,dz));
      double eig=.5-std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
      if(!(eig>0) || !std::isfinite(dr) || !std::isfinite(dz))err=std::numeric_limits<double>::infinity();
     }
    }else{
     for(int p=0;p<4;p++){
      int e=4*i+p;err=std::max(err,std::abs(prop[e]-cur[e]));mx=std::max(mx,std::abs(cur[e]));
      if(!std::isfinite(prop[e]))err=std::numeric_limits<double>::infinity();
     }
     if(d.domain==0){double h=d.field[i]-d.diag[i]*x[i];
      for(int p=0;p<4;p++){int j=d.nb[4*i+p];if(j>=0)h+=x[j];}resvec[i]=h;er2+=h*h;
     }
    }
   }
   local1[g]=err;local2[g]=mx;local3[g]=er2;
  }
 }
 void correction(double cap){
  #pragma omp for schedule(static)
  for(int g=0;g<d.G;g++){
   double dot=0;const double be=beta[g];
   for(int it=d.offset[g];it<d.offset[g+1];it++){
    int i=d.order[it];
    if(d.domain==0){
     double old=previous[i],dx=.45*resvec[i]/d.diag[i]+(has?be*(x[i]-old):0.);
     if(has)dot+=resvec[i]*(x[i]+dx-old);previous[i]=x[i];double s=d.field[i];
     for(int p=0;p<4;p++){int e=4*i+p;if(d.nb[e]>=0)cur[e]+=d.P[i]*d.p[e]/(d.P[i]-d.prior[i])*dx;s+=cur[e];}
     total[i]=s;
    }else if(d.domain==1){
     double s=d.field[i];
     for(int p=0;p<4;p++){int e=4*i+p;double bar=cur[e],r=prop[e]-bar,old=previous[e];
      double v=bar+.45*r+(has?be*(bar-old):0.);if(has)dot+=r*(v-old);previous[e]=bar;cur[e]=v;s+=v;
     }total[i]=s;
    }else{
     for(int p=0;p<4;p++){
      int e=12*i+3*p;double bar[3],pr[3],v[3];logvec(cur+e,bar);logvec(prop+e,pr);
      for(int a=0;a<3;a++){double r=pr[a]-bar[a],old=previous[e+a];v[a]=bar[a]+.45*r+(has?be*(bar[a]-old):0.);
       if(has)dot+=2*r*(v[a]-old);previous[e+a]=bar[a];
      }expvec(v,cur+e);
     }
    }
   }
   if(dot<0){clock[g]=1;beta[g]=0;}else{double t=(1+std::sqrt(1+4*clock[g]*clock[g]))/2;beta[g]=std::min(cap,(clock[g]-1)/t);clock[g]=t;}
  }
 }
};
struct Result{int cycles=0,sweeps=0,corrections=0,evals=0;double residual=0,message_residual=0,wall=0,cpu=0,correction_wall=0;bool converged=false;V values,history;};
Result solve(const Data& d,int method,int threads,int maxcycles,double tolerance,int exactcycles=0,const std::string& init=""){
 State s(d);Result result;bool done=false;double rs=1,rm=1,bn=0,cstart=0;
 for(double v:d.field)bn+=v*v;bn=std::sqrt(bn);
 if(!init.empty()){
  std::ifstream f(init,std::ios::binary);f.read((char*)s.cur,s.a.size()*8);if(!f)throw std::runtime_error("bad initial state");
  if(d.domain!=2)for(int i=0;i<d.N;i++){s.total[i]=d.field[i];for(int p=0;p<4;p++)s.total[i]+=s.cur[4*i+p];}
 }
 double start=omp_get_wtime(),cput=cpu_seconds();
 #pragma omp parallel num_threads(threads) shared(done,result,rs,rm,cstart,s)
 {
  for(int cyc=0;cyc<maxcycles;cyc++){
   for(int step=(method==0 && cyc>0)?1:0;step<8;step++){
    s.sweep(s.cur,s.total,s.nxt,s.totalNext);
    #pragma omp single
    {std::swap(s.cur,s.nxt);std::swap(s.total,s.totalNext);}
   }
   s.sweep(s.cur,s.total,s.prop,s.totalProp);s.residuals();
   #pragma omp single
   {
    double em=0,mm=0,ss=0;for(int g=0;g<d.G;g++){em=std::max(em,s.local1[g]);mm=std::max(mm,s.local2[g]);ss+=s.local3[g];}
    rm=em/(1+mm);rs=d.domain==0?std::sqrt(ss)/bn:rm;
    result.cycles++;result.sweeps+=8;result.evals+=(method==0 && cyc>0)?8:9;result.residual=rs;result.message_residual=rm;
    result.history.insert(result.history.end(),{double(result.sweeps),double(result.evals),double(result.corrections),rs,rm});
    done=(!std::isfinite(rs)||!std::isfinite(rm)||(!exactcycles && rs<=tolerance && rm<=tolerance));
    if(exactcycles && cyc+1>=exactcycles)done=true;
    if(cyc+1>=maxcycles)done=true;
    cstart=omp_get_wtime();
   }
   if(done)break;
   if(method){
    s.correction(method==2?0.:.95);
    #pragma omp single
    {s.has=true;result.corrections++;result.correction_wall+=omp_get_wtime()-cstart;}
   }else{
    // Accept the already-computed stopping proposal as the next cycle's first sweep.
    #pragma omp single
    {std::swap(s.cur,s.prop);std::swap(s.total,s.totalProp);}
   }
  }
 }
 result.wall=omp_get_wtime()-start;result.cpu=cpu_seconds()-cput;result.converged=result.residual<=tolerance && result.message_residual<=tolerance;
 result.values.assign(s.cur,s.cur+s.a.size());return result;
}
int main(int argc,char**argv){try{
 if(argc<5){std::cerr<<"usage: benchmark fixture method(0=BP,1=IBP,2=no-momentum) threads maxcycles [exactcycles=0] [output-prefix=-] [initial-state=-]\n";return 2;}
 Data d(argv[1]);int method=std::stoi(argv[2]),nth=std::stoi(argv[3]),maxcycles=std::stoi(argv[4]);int ex=argc>5?std::stoi(argv[5]):0;
 std::string prefix=argc>6?argv[6]:"-",init=argc>7?argv[7]:"-";if(init=="-")init="";
 omp_set_dynamic(0);double tol=d.domain==0?1e-8:1e-10;
 // Native-code/thread-pool warm-up; zero state is re-created afterwards. Not timed as solve.
 auto warm=solve(d,method,nth,2,-1.,2);
 Result r=solve(d,method,nth,maxcycles,tol,ex,init);
 if(prefix!="-"){std::ofstream f(prefix+".state",std::ios::binary);f.write((char*)r.values.data(),8*r.values.size());std::ofstream h(prefix+".history",std::ios::binary);h.write((char*)r.history.data(),8*r.history.size());}
 std::cout<<std::setprecision(17)<<"{\"domain\":"<<d.domain<<",\"L\":"<<d.L<<",\"N\":"<<d.N<<",\"seed\":"<<d.seed
 <<",\"method\":"<<method<<",\"threads\":"<<nth<<",\"cycles\":"<<r.cycles<<",\"sweeps\":"<<r.sweeps<<",\"evals\":"<<r.evals<<",\"corrections\":"<<r.corrections
 <<",\"residual\":"<<r.residual<<",\"message_residual\":"<<r.message_residual<<",\"seconds\":"<<r.wall<<",\"cpu_seconds\":"<<r.cpu
 <<",\"correction_seconds\":"<<r.correction_wall<<",\"converged\":"<<(r.converged?"true":"false")<<"}\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
