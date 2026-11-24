//
//  vorticity.c
//  
//
//  Created by Shawn Shadden.
//  Copyright 2010 Illinois Institute of Technology. All rights reserved.
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "globals.h"
#include "io.h"
#include "macros.h"
#include "memory.h"
#include "mesh.h"
#include "vorticity.h"
#include "structs.h"
#include "tracers.h"
#include "velocity.h"


void AllocateVorticityFieldData(void) {
	/*
	 *	Allocates memory for vorticity field data for 2 time frames (moving buffer of data)
	 */
	
	printf("\n  Allocating memory for vorticity data...");
    fflush(stdout);
	if (Data_MeshType == CARTESIAN) {
		Vel_CartVorArray_wx = Calloc4D(Vel_CartVorArray_wx, 2, Vel_CartMesh.XRes - 1, Vel_CartMesh.YRes - 1, Vel_CartMesh.ZRes - 1);
		Vel_CartVorArray_wy = Calloc4D(Vel_CartVorArray_wy, 2, Vel_CartMesh.XRes - 1, Vel_CartMesh.YRes - 1, Vel_CartMesh.ZRes - 1);
		Vel_CartVorArray_wz  = Calloc4D(Vel_CartVorArray_wz, 2, Vel_CartMesh.XRes - 1, Vel_CartMesh.YRes - 1, Vel_CartMesh.ZRes - 1);
	}
	else {
		Vel_UnstructVorArray_wx = Calloc2D(Vel_UnstructVorArray_wx, 2, Vel_MeshNumElements);
		Vel_UnstructVorArray_wy = Calloc2D(Vel_UnstructVorArray_wy, 2, Vel_MeshNumElements);
		Vel_UnstructVorArray_wz  = Calloc2D(Vel_UnstructVorArray_wz, 2, Vel_MeshNumElements);
	}
	printf("OK!\n");
    fflush(stdout);
	
}

void FreeVorticityData(void) {
	/*** 
	 Deallocate memory for vorticity field for 2 time frames (moving buffer of data)
	 ***/
	if (Data_MeshType == CARTESIAN) {
		Free4D(Vel_CartVorArray_wx, 2, Vel_CartMesh.XRes - 1, Vel_CartMesh.YRes - 1);
		Free4D(Vel_CartVorArray_wy, 2, Vel_CartMesh.XRes - 1, Vel_CartMesh.YRes - 1);
		Free4D(Vel_CartVorArray_wz, 2, Vel_CartMesh.XRes - 1, Vel_CartMesh.YRes - 1);
	}
	else {
		Free2D(Vel_UnstructVorArray_wx, 2);
		Free2D(Vel_UnstructVorArray_wy, 2);
		Free2D(Vel_UnstructVorArray_wz, 2);		
	}
}

void LoadVorticityDataFrame(int frame) {
    if(Data_MeshType == CARTESIAN)  
        LoadCartVorticityDataFrame(frame);
    else if(Data_MeshType == UNSTRUCTURED)
        LoadUnstructVorticityDataFrame(frame);
    else
        FatalError("Unknown value for Data_MeshType in LoadVorticityDataFrame()");
}


void LoadCartVorticityDataFrame(int frame) {
	
	int i, j, k, ModVal, slot1, slot2;
	double timestamp;
	char Data_BinFilePath[LONGSTRING];
	FILE *Data_BinFileID;
	
	if(Int_TimeDirection > 0) {
		slot1 = 0;
		slot2 = 1;
	}    
	else {
		slot1 = 1; 
		slot2 = 0;
	}
	
	ModVal = fmod(frame, Data_TRes - 1); 
	if(ModVal < 0) {
		ModVal += Data_TRes - 1;
	}
	
	if(Vel_CartVorArray_wx == NULL || Vel_CartVorArray_wy == NULL || Vel_CartVorArray_wz == NULL)
		AllocateVorticityFieldData();

	/* Swap slot pointers in O(1) instead of iterating over all (i,j) pairs */
	{
		double ***tmp;
		tmp = Vel_CartVorArray_wx[slot1]; Vel_CartVorArray_wx[slot1] = Vel_CartVorArray_wx[slot2]; Vel_CartVorArray_wx[slot2] = tmp;
		tmp = Vel_CartVorArray_wy[slot1]; Vel_CartVorArray_wy[slot1] = Vel_CartVorArray_wy[slot2]; Vel_CartVorArray_wy[slot2] = tmp;
		tmp = Vel_CartVorArray_wz[slot1]; Vel_CartVorArray_wz[slot1] = Vel_CartVorArray_wz[slot2]; Vel_CartVorArray_wz[slot2] = tmp;
	}

	/*** Read in new time slice of data ***/
	/* Open file */
	sprintf(Data_BinFilePath, "%s%s_vorticity.%d.bin", Path_Data, Data_InFilePrefix, Data_SuffixTMin + ModVal * Data_SuffixTDelta);
	if((Data_BinFileID = fopen(Data_BinFilePath, "rb")) == NULL) 
		FatalError("Could not open file %s", Data_BinFilePath);
	/* Read time stamp */
	if(fread(&timestamp, sizeof(double), 1, Data_BinFileID) < 1) 
		FatalError("Could not read time stamp from file %s", Data_BinFilePath);
	printf("\nLoading vorticity data from %s (time stamp = %g)...", Data_BinFilePath, timestamp);
	fflush(stdout);
	/* Read vorticity data using bulk row reads */
	{
		int xdim = Vel_CartMesh.XRes - 1;
		double *rowbuf;
		if((rowbuf = (double *)malloc(xdim * 3 * sizeof(double))) == NULL)
			FatalError("Malloc failed for rowbuf in LoadCartVorticityDataFrame");
		for(k = 0; k < Vel_CartMesh.ZRes - 1; k++) {
			for(j = 0; j < Vel_CartMesh.YRes - 1; j++) {
				if((int)fread(rowbuf, sizeof(double), xdim * 3, Data_BinFileID) < xdim * 3)
					FatalError("Could not load vorticity row k=%d j=%d from %s", k, j, Data_BinFilePath);
				for(i = 0; i < xdim; i++) {
					Vel_CartVorArray_wx[slot2][i][j][k] = rowbuf[i * 3    ];
					Vel_CartVorArray_wy[slot2][i][j][k] = rowbuf[i * 3 + 1];
					Vel_CartVorArray_wz[slot2][i][j][k] = rowbuf[i * 3 + 2];
				}
			}
		}
		free(rowbuf);
	}
	/* Close file */
	fclose(Data_BinFileID);
	
	printf("OK!\n");
	fflush(stdout);
	
}


/* Vorticity is fread in chunks of 65536 elements instead of one (wx,wy,wz)
 * triple per call.
 */
void LoadUnstructVorticityDataFrame(int frame) {
	
	int i, ModVal, slot1, slot2;
	double *tempptr_U, *tempptr_V, *tempptr_W, timestamp;
	char Data_BinFilePath[LONGSTRING];
	FILE *Data_BinFileID;
	
	if(Int_TimeDirection > 0) {
		slot1 = 0;
		slot2 = 1;
	}    
	else {
		slot1 = 1; 
		slot2 = 0;
	}
	
	ModVal = fmod(frame, Data_TRes - 1); 
	if(ModVal < 0) {
		ModVal += Data_TRes - 1;
	}
	
	/* Allocate memory for velocity data if needed */
	if(Vel_UnstructVorArray_wx == NULL || Vel_UnstructVorArray_wy == NULL || Vel_UnstructVorArray_wz == NULL) 
		AllocateVorticityFieldData();
	
	/* Open binary velocity file for reading */
	sprintf(Data_BinFilePath, "%s%s_vorticity.%d.bin", Path_Data, Data_InFilePrefix, Data_SuffixTMin + ModVal * Data_SuffixTDelta);
	if((Data_BinFileID = fopen(Data_BinFilePath, "rb")) == NULL) 
		FatalError("Could not open file %s", Data_BinFilePath);
	
	/* Read time stamp */
	if(fread(&timestamp, sizeof(double), 1, Data_BinFileID) < 1) 
		FatalError("Could not read time stamp from file %s", Data_BinFilePath);
	
	printf("\nLoading vorticity data from %s (time stamp = %g)...", Data_BinFilePath, timestamp);
	fflush(stdout);
    
	/* Virtual data shift */
	tempptr_U = Vel_UnstructVorArray_wx[slot1];
	tempptr_V = Vel_UnstructVorArray_wy[slot1];
	tempptr_W = Vel_UnstructVorArray_wz[slot1];
	Vel_UnstructVorArray_wx[slot1] = Vel_UnstructVorArray_wx[slot2];
	Vel_UnstructVorArray_wy[slot1] = Vel_UnstructVorArray_wy[slot2]; 
	Vel_UnstructVorArray_wz[slot1] = Vel_UnstructVorArray_wz[slot2];
	Vel_UnstructVorArray_wx[slot2] = tempptr_U;
	Vel_UnstructVorArray_wy[slot2] = tempptr_V;
	Vel_UnstructVorArray_wz[slot2] = tempptr_W;
	
	/* Read vorticity data in fixed-size chunks to bound peak memory. */
	{
#define VOR_READ_CHUNK_ELEMS 65536
		double uvw_buf[VOR_READ_CHUNK_ELEMS * 3];
		int remaining = Vel_MeshNumElements, base = 0, chunk, ci;
		while(remaining > 0) {
			chunk = (remaining < VOR_READ_CHUNK_ELEMS) ? remaining : VOR_READ_CHUNK_ELEMS;
			if(fread(uvw_buf, sizeof(double), (size_t)chunk * 3, Data_BinFileID) < (size_t)chunk * 3)
				FatalError("Could not read vorticity data from file %s", Data_BinFilePath);
			for(ci = 0; ci < chunk; ci++) {
				Vel_UnstructVorArray_wx[slot2][base + ci] = uvw_buf[ci * 3    ];
				Vel_UnstructVorArray_wy[slot2][base + ci] = uvw_buf[ci * 3 + 1];
				Vel_UnstructVorArray_wz[slot2][base + ci] = uvw_buf[ci * 3 + 2];
			}
			base      += chunk;
			remaining -= chunk;
		}
#undef VOR_READ_CHUNK_ELEMS
	}
        
	fclose(Data_BinFileID);
	printf("OK!\n");
	fflush(stdout);
	
}



void SetVorticity(double tc, LagrangianPoint *pt) {
	
	if(Data_MeshType == CARTESIAN)
		SetVorticity_Cartesian(tc, pt);
	else if(Data_MeshType == UNSTRUCTURED) 
		SetVorticity_Unstructured(tc, pt);
	else if(Data_MeshType == ANALYTIC)
		FatalError("Data_MeshType == ANALYTIC in SetVorticity() unsupported");
	else 
		FatalError("Unrecognized value for Data_MeshType in SetVorticity()");
	
}


void SetVorticity_Cartesian(double time, LagrangianPoint *pt) {
	
	double tloc;
	int    i, j, k;
	
	if(TestOutsideDomain(pt->X))
		FatalError("Attempting to interpolate vorticity at point that has left the domain");
	
	tloc = (time - Data_LoadedTMin) / (Data_LoadedTMax - Data_LoadedTMin);
	assert((tloc < (1 + TINY)) && (tloc > (0 - TINY)));
	
	i = (int)floor((pt->X[0] - Vel_CartMesh.XMin) / Vel_CartMesh.XDelta);
	assert((i >= 0) && (i <= (Vel_CartMesh.XRes - 1)));
	if(i == (Vel_CartMesh.XRes - 1)) 
		i = Vel_CartMesh.XRes - 2;
	j = (int)floor((pt->X[1] - Vel_CartMesh.YMin) / Vel_CartMesh.YDelta);
	assert((j >= 0) && (j <= (Vel_CartMesh.XRes - 1)));
	if(j == (Vel_CartMesh.YRes - 1)) 
		j = Vel_CartMesh.YRes - 2;
	if(Dimensions == 3) {
		k = (int)floor((pt->X[2] - Vel_CartMesh.ZMin) / Vel_CartMesh.ZDelta);
		assert((k >= 0) && (k <= (Vel_CartMesh.ZRes - 1)));
		if(k == (Vel_CartMesh.ZRes - 1)) 
			k = Vel_CartMesh.ZRes - 2;
	}
	else
		k = 0;
    
	pt->vorticity[0] = ((1 - tloc) * Vel_CartVorArray_wx[0][i][j][k] + tloc * Vel_CartVorArray_wx[1][i][j][k]);
    pt->vorticity[1] = ((1 - tloc) * Vel_CartVorArray_wy[0][i][j][k] + tloc * Vel_CartVorArray_wy[1][i][j][k]);
    pt->vorticity[2] = ((1 - tloc) * Vel_CartVorArray_wz[0][i][j][k] + tloc * Vel_CartVorArray_wz[1][i][j][k]);
	
}


void SetVorticity_Unstructured(double time, LagrangianPoint *pt) {
	
	double tloc;
	
	pt->ElementIndex = Get_Element_Local_Search(pt->X, pt->ElementIndex);
	
	if(pt->ElementIndex == -1) 
		FatalError("Attempting to interpolate vorticity at point with pt->ElementIndex = -1");
	
	/* Set tloc defining where in between loaded data to interpolate in time */
	tloc = (time - Data_LoadedTMin) / (Data_LoadedTMax - Data_LoadedTMin);
	if((tloc > (1 + TINY)) || (tloc < (0 - TINY))) 
		FatalError("tloc = %f and must be between 0 and 1", tloc);
	
	/* Interpolate vorticity at element in time */
	pt->vorticity[0] = (1 - tloc) * Vel_UnstructVorArray_wx[0][pt->ElementIndex] + tloc * Vel_UnstructVorArray_wx[1][pt->ElementIndex];
    pt->vorticity[1] = (1 - tloc) * Vel_UnstructVorArray_wy[0][pt->ElementIndex] + tloc * Vel_UnstructVorArray_wy[1][pt->ElementIndex];
    pt->vorticity[2] = (1 - tloc) * Vel_UnstructVorArray_wz[0][pt->ElementIndex] + tloc * Vel_UnstructVorArray_wz[1][pt->ElementIndex];
    
    /* printf("vorticity = (%f, %f, %f)\n", pt->vorticity[0], pt->vorticity[1], pt->vorticity[2]);*/
    
	
}
