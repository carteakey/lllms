#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(){
    size_t N = 64*1024*1024/sizeof(size_t); // 64 MiB
    size_t *buf = malloc(N*sizeof(size_t));
    for(size_t i=0;i<N;i++) buf[i]=i;
    srand(42);
    for(size_t i=N-1;i>0;i--){ size_t j=rand()% (i+1); size_t t=buf[i];buf[i]=buf[j];buf[j]=t; }
    size_t p=0; volatile size_t sink=0;
    struct timespec a,b; int iters=8, hops=8000000;
    double best=1e9;
    for(int it=0; it<iters; it++){
        clock_gettime(CLOCK_MONOTONIC,&a);
        for(long h=0;h<hops;h++){ p=buf[p]; sink+=p; }
        clock_gettime(CLOCK_MONOTONIC,&b);
        double ns=(b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec);
        if(ns/hops<best) best=ns/hops;
    }
    printf("random 64MiB chase: %.1f ns/hop (~%.0f Mhops/s) | best of %d\n", best, 1000/best, iters);
    return sink>0?0:1;
}
