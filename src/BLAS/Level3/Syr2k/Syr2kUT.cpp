/*
   Copyright 2009-2010 Jack Poulson

   This file is part of Elemental.

   Elemental is free software: you can redistribute it and/or modify it under
   the terms of the GNU Lesser General Public License as published by the
   Free Software Foundation; either version 3 of the License, or 
   (at your option) any later version.

   Elemental is distributed in the hope that it will be useful, but 
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with Elemental. If not, see <http://www.gnu.org/licenses/>.
*/
#include "Elemental/BLASInternal.hpp"
using namespace std;
using namespace Elemental;

template<typename T>
void
Elemental::BLAS::Internal::Syr2kUT
( const T alpha, const DistMatrix<T,MC,MR>& A,
                 const DistMatrix<T,MC,MR>& B,
  const T beta,        DistMatrix<T,MC,MR>& C )
{
#ifndef RELEASE
    PushCallStack("BLAS::Internal::Syr2kUT");
    if( A.GetGrid() != B.GetGrid() || B.GetGrid() != C.GetGrid() )
        throw "{A,B,C} must be distributed over the same grid.";
    if( A.Width() != C.Height() || 
        A.Width() != C.Width()  ||
        B.Width() != C.Height() ||
        B.Width() != C.Width()  ||
        A.Height() != B.Height()  )
    {
        ostringstream msg;
        msg << "Nonconformal Syr2kUT:" << endl
            << "  A ~ " << A.Height() << " x " << A.Width() << endl
            << "  B ~ " << B.Height() << " x " << B.Width() << endl
            << "  C ~ " << C.Height() << " x " << C.Width() << endl;
        throw msg.str();
    }
#endif
    const Grid& grid = A.GetGrid();

    // Matrix views
    DistMatrix<T,MC,MR> AT(grid),  A0(grid),
                        AB(grid),  A1(grid),
                                   A2(grid);

    DistMatrix<T,MC,MR> BT(grid),  B0(grid),
                        BB(grid),  B1(grid),
                                   B2(grid);

    // Temporary distributions
    DistMatrix<T,Star,MC> A1_Star_MC(grid);
    DistMatrix<T,Star,MR> A1_Star_MR(grid);
    DistMatrix<T,Star,MC> B1_Star_MC(grid);
    DistMatrix<T,Star,MR> B1_Star_MR(grid);

    // Start the algorithm
    BLAS::Scal( beta, C );
    LockedPartitionDown( A, AT, 
                            AB );
    LockedPartitionDown( B, BT,
                            BB );
    while( AB.Height() > 0 )
    {
        LockedRepartitionDown( AT,  A0,
                              /**/ /**/
                                    A1,
                               AB,  A2 );

        LockedRepartitionDown( BT,  B0,
                              /**/ /**/
                                    B1,
                               BB,  B2 );

        A1_Star_MC.AlignWith( C );
        A1_Star_MR.AlignWith( C );
        B1_Star_MC.AlignWith( C );
        B1_Star_MR.AlignWith( C );
        //--------------------------------------------------------------------//
        A1_Star_MR = A1;
        A1_Star_MC = A1_Star_MR;
        B1_Star_MR = B1;
        B1_Star_MC = B1_Star_MR;

        BLAS::Internal::Syr2kUTUpdate
        ( alpha, A1_Star_MC, A1_Star_MR, 
                 B1_Star_MC, B1_Star_MR, (T)1, C ); 
        //--------------------------------------------------------------------//
        A1_Star_MC.FreeConstraints();
        A1_Star_MR.FreeConstraints();
        B1_Star_MC.FreeConstraints();
        B1_Star_MR.FreeConstraints();

        SlideLockedPartitionDown( AT,  A0,
                                       A1,
                                 /**/ /**/
                                  AB,  A2 );

        SlideLockedPartitionDown( BT,  B0,
                                       B1,
                                 /**/ /**/
                                  BB,  B2 );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
Elemental::BLAS::Internal::Syr2kUTUpdate
( const T alpha, const DistMatrix<T,Star,MC>& A_Star_MC,
                 const DistMatrix<T,Star,MR>& A_Star_MR,
                 const DistMatrix<T,Star,MC>& B_Star_MC,
                 const DistMatrix<T,Star,MR>& B_Star_MR,
  const T beta,        DistMatrix<T,MC,  MR>& C         )
{
#ifndef RELEASE
    PushCallStack("BLAS::Internal::SyrkUTUpdate");
    if( A_Star_MC.GetGrid() != A_Star_MR.GetGrid() ||
        A_Star_MR.GetGrid() != B_Star_MC.GetGrid() ||
        B_Star_MC.GetGrid() != B_Star_MR.GetGrid() ||
        B_Star_MR.GetGrid() != C.GetGrid()           )
    {
        throw "{A,B,C} must be distributed over the same grid.";
    }
    if( A_Star_MC.Width() != C.Height() ||
        A_Star_MR.Width() != C.Width()  ||
        B_Star_MC.Width() != C.Height() ||
        B_Star_MR.Width() != C.Width()  ||
        A_Star_MC.Height() != A_Star_MR.Height() ||
        A_Star_MC.Width()  != A_Star_MR.Width()  ||  
        B_Star_MC.Height() != B_Star_MR.Height() ||
        B_Star_MC.Width()  != B_Star_MR.Width()     )
    {
        ostringstream msg;
        msg << "Nonconformal Syr2kUTUpdate: " << endl
            << "  A[* ,MC] ~ " << A_Star_MC.Height() << " x "
                               << A_Star_MC.Width()  << endl
            << "  A[* ,MR] ~ " << A_Star_MR.Height() << " x "
                               << A_Star_MR.Width()  << endl
            << "  B[* ,MC] ~ " << B_Star_MC.Height() << " x "
                               << B_Star_MC.Width()  << endl
            << "  B[* ,MR] ~ " << B_Star_MR.Height() << " x "
                               << B_Star_MR.Width()  << endl
            << "  C[MC,MR] ~ " << C.Height() << " x " << C.Width() << endl;
        throw msg.str();
    }
    if( A_Star_MC.RowAlignment() != C.ColAlignment() ||
        A_Star_MR.RowAlignment() != C.RowAlignment() ||  
        B_Star_MC.RowAlignment() != C.ColAlignment() ||
        B_Star_MR.RowAlignment() != C.RowAlignment()    )
    {
        ostringstream msg;
        msg << "Misaligned Syr2kUTUpdate: " << endl
            << "  A[* ,MC] ~ " << A_Star_MC.RowAlignment() << endl
            << "  A[* ,MR] ~ " << A_Star_MR.RowAlignment() << endl
            << "  B[* ,MC] ~ " << B_Star_MC.RowAlignment() << endl
            << "  B[* ,MR] ~ " << B_Star_MR.RowAlignment() << endl
            << "  C[MC,MR] ~ " << C.ColAlignment() << " , " <<
                                  C.RowAlignment() << endl;
        throw msg.str();
    }
#endif
    const Grid& grid = C.GetGrid();

    // Matrix views
    DistMatrix<T,Star,MC> 
        AL_Star_MC(grid), AR_Star_MC(grid),
        A0_Star_MC(grid), A1_Star_MC(grid), A2_Star_MC(grid);

    DistMatrix<T,Star,MR> 
        AL_Star_MR(grid), AR_Star_MR(grid),
        A0_Star_MR(grid), A1_Star_MR(grid), A2_Star_MR(grid);

    DistMatrix<T,Star,MC> 
        BL_Star_MC(grid), BR_Star_MC(grid),
        B0_Star_MC(grid), B1_Star_MC(grid), B2_Star_MC(grid);

    DistMatrix<T,Star,MR> 
        BL_Star_MR(grid), BR_Star_MR(grid),
        B0_Star_MR(grid), B1_Star_MR(grid), B2_Star_MR(grid);

    DistMatrix<T,MC,MR> 
        CTL(grid), CTR(grid),  C00(grid), C01(grid), C02(grid), 
        CBL(grid), CBR(grid),  C10(grid), C11(grid), C12(grid),
                               C20(grid), C21(grid), C22(grid);

    DistMatrix<T,MC,MR> D11(grid);

    // We want our local gemms to be of width blocksize, and so we will 
    // temporarily change to c times the current blocksize
    PushBlocksizeStack( grid.Width()*Blocksize() );

    // Start the algorithm
    BLAS::Scal( beta, C );
    LockedPartitionLeft( A_Star_MC, AL_Star_MC, AR_Star_MC );
    LockedPartitionLeft( A_Star_MR, AL_Star_MR, AR_Star_MR );
    LockedPartitionLeft( B_Star_MC, BL_Star_MC, BR_Star_MC );
    LockedPartitionLeft( B_Star_MR, BL_Star_MR, BR_Star_MR );
    PartitionUpDiagonal( C, CTL, CTR,
                            CBL, CBR );
    while( AL_Star_MC.Width() > 0 )
    {
        LockedRepartitionLeft( AL_Star_MC,             /**/ AR_Star_MC,
                               A0_Star_MC, A1_Star_MC, /**/ A2_Star_MC );

        LockedRepartitionLeft( AL_Star_MR,             /**/ AR_Star_MR,
                               A0_Star_MR, A1_Star_MR, /**/ A2_Star_MR );
        
        LockedRepartitionLeft( BL_Star_MC,             /**/ BR_Star_MC,
                               B0_Star_MC, B1_Star_MC, /**/ B2_Star_MC );

        LockedRepartitionLeft( BL_Star_MR,             /**/ BR_Star_MR,
                               B0_Star_MR, B1_Star_MR, /**/ B2_Star_MR );

        RepartitionUpDiagonal( CTL, /**/ CTR,  C00, C01, /**/ C02,
                                    /**/       C10, C11, /**/ C12,
                              /*************/ /******************/
                               CBL, /**/ CBR,  C20, C21, /**/ C22 );

        D11.AlignWith( C11 );
        D11.ResizeTo( C11.Height(), C11.Width() );
        //--------------------------------------------------------------------//
        BLAS::Gemm( Transpose, Normal,
                    alpha, A1_Star_MC.LockedLocalMatrix(),
                           B1_Star_MR.LockedLocalMatrix(),
                    (T)0,  D11.LocalMatrix()              );
        BLAS::Gemm( Transpose, Normal,
                    alpha, A0_Star_MC.LockedLocalMatrix(),
                           B1_Star_MR.LockedLocalMatrix(),
                    (T)1,  C01.LocalMatrix()              );

        BLAS::Gemm( Transpose, Normal,
                    alpha, B0_Star_MC.LockedLocalMatrix(),
                           A1_Star_MR.LockedLocalMatrix(),
                    (T)1,  C01.LocalMatrix()              );
        BLAS::Gemm( Transpose, Normal,
                    alpha, B1_Star_MC.LockedLocalMatrix(),
                           A1_Star_MR.LockedLocalMatrix(),
                    (T)1,  D11.LocalMatrix()              );

        D11.MakeTrapezoidal( Left, Upper );
        BLAS::Axpy( (T)1, D11, C11 );
        //--------------------------------------------------------------------//
        D11.FreeConstraints();
        
        SlideLockedPartitionLeft( AL_Star_MC, /**/ AR_Star_MC,
                                  A0_Star_MC, /**/ A1_Star_MC, A2_Star_MC );

        SlideLockedPartitionLeft( AL_Star_MR, /**/ AR_Star_MR,
                                  A0_Star_MR, /**/ A1_Star_MR, A2_Star_MR );

        SlideLockedPartitionLeft( BL_Star_MC, /**/ BR_Star_MC,
                                  B0_Star_MC, /**/ B1_Star_MC, B2_Star_MC );

        SlideLockedPartitionLeft( BL_Star_MR, /**/ BR_Star_MR,
                                  B0_Star_MR, /**/ B1_Star_MR, B2_Star_MR );

        SlidePartitionUpDiagonal( CTL, /**/ CTR,  C00, /**/ C01, C02,
                                 /*************/ /******************/
                                       /**/       C10, /**/ C11, C12,
                                  CBL, /**/ CBR,  C20, /**/ C21, C22 );
    }

    PopBlocksizeStack();

#ifndef RELEASE
    PopCallStack();
#endif
}

template void Elemental::BLAS::Internal::Syr2kUT
( const float alpha, const DistMatrix<float,MC,MR>& A,
                     const DistMatrix<float,MC,MR>& B,
  const float beta,        DistMatrix<float,MC,MR>& C );

template void Elemental::BLAS::Internal::Syr2kUT
( const double alpha, const DistMatrix<double,MC,MR>& A,
                      const DistMatrix<double,MC,MR>& B,
  const double beta,        DistMatrix<double,MC,MR>& C );

#ifndef WITHOUT_COMPLEX
template void Elemental::BLAS::Internal::Syr2kUT
( const scomplex alpha, const DistMatrix<scomplex,MC,MR>& A,
                        const DistMatrix<scomplex,MC,MR>& B,
  const scomplex beta,        DistMatrix<scomplex,MC,MR>& C );

template void Elemental::BLAS::Internal::Syr2kUT
( const dcomplex alpha, const DistMatrix<dcomplex,MC,MR>& A,
                        const DistMatrix<dcomplex,MC,MR>& B,
  const dcomplex beta,        DistMatrix<dcomplex,MC,MR>& C );
#endif

