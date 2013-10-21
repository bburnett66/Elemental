/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ELEM_CORE_DISTMATRIX_MC_STAR_DECL_HPP
#define ELEM_CORE_DISTMATRIX_MC_STAR_DECL_HPP

namespace elem {

// Partial specialization to A[MC,* ].
//
// The rows of these distributed matrices will be replicated on all 
// processes (*), and the columns will be distributed like "Matrix Columns" 
// (MC). Thus the columns will be distributed among columns of the process
// grid.
template<typename T>
class DistMatrix<T,MC,STAR> : public AbstractDistMatrix<T>
{
public:
    typedef AbstractDistMatrix<T> admType;
    typedef DistMatrix<T,MC,STAR> type;

    // Create a 0 x 0 distributed matrix
    DistMatrix( const elem::Grid& g=DefaultGrid() );

    // Create a height x width distributed matrix
    DistMatrix
    ( Int height, Int width, const elem::Grid& g=DefaultGrid() );

    // Create a height x width distributed matrix with specified alignments
    DistMatrix
    ( Int height, Int width, Int colAlignment, const elem::Grid& g );

    // Create a height x width distributed matrix with specified alignments
    // and leading dimension
    DistMatrix
    ( Int height, Int width, 
      Int colAlignment, Int ldim, const elem::Grid& g );

    // View a constant distributed matrix's buffer
    DistMatrix
    ( Int height, Int width, Int colAlignment, 
      const T* buffer, Int ldim, const elem::Grid& g );

    // View a mutable distributed matrix's buffer
    DistMatrix
    ( Int height, Int width, Int colAlignment,
      T* buffer, Int ldim, const elem::Grid& g );

    // Create a copy of distributed matrix A
    DistMatrix( const type& A );
    template<Distribution U,Distribution V>
    DistMatrix( const DistMatrix<T,U,V>& A );

    ~DistMatrix();

#ifndef SWIG
    // Move constructor
    DistMatrix( type&& A );
    // Move assignment
    type& operator=( type&& A );
#endif

    const type& operator=( const DistMatrix<T,MC,  MR  >& A );
    const type& operator=( const DistMatrix<T,MC,  STAR>& A );
    const type& operator=( const DistMatrix<T,STAR,MR  >& A );
    const type& operator=( const DistMatrix<T,MD,  STAR>& A );
    const type& operator=( const DistMatrix<T,STAR,MD  >& A );
    const type& operator=( const DistMatrix<T,MR,  MC  >& A );
    const type& operator=( const DistMatrix<T,MR,  STAR>& A );
    const type& operator=( const DistMatrix<T,STAR,MC  >& A );
    const type& operator=( const DistMatrix<T,VC,  STAR>& A );
    const type& operator=( const DistMatrix<T,STAR,VC  >& A );
    const type& operator=( const DistMatrix<T,VR,  STAR>& A );
    const type& operator=( const DistMatrix<T,STAR,VR  >& A );
    const type& operator=( const DistMatrix<T,STAR,STAR>& A );
    const type& operator=( const DistMatrix<T,CIRC,CIRC>& A );

    //------------------------------------------------------------------------//
    // Overrides of AbstractDistMatrix                                        //
    //------------------------------------------------------------------------//

    //
    // Non-collective routines
    //

    virtual elem::DistData DistData() const;
    virtual mpi::Comm DistComm() const;
    virtual mpi::Comm CrossComm() const;
    virtual mpi::Comm RedundantComm() const;
    virtual mpi::Comm ColComm() const;
    virtual mpi::Comm RowComm() const;
    virtual Int RowStride() const;
    virtual Int ColStride() const;

    //
    // Collective routines
    //

    // Distribution alignment
    virtual void AlignWith( const admType& A );
    virtual void AlignWith( const elem::DistData& data );
    virtual void AlignColsWith( const admType& A );
    virtual void AlignColsWith( const elem::DistData& data );

    //------------------------------------------------------------------------//
    // Routines specific to [MC,* ] distribution                              //
    //------------------------------------------------------------------------//

    //
    // Collective routines
    //

    void GetDiagonal( type& d, Int offset=0 ) const;
    void GetDiagonal( DistMatrix<T,STAR,MC>& d, Int offset=0 ) const;
    void GetRealPartOfDiagonal
    ( DistMatrix<BASE(T),MC,STAR>& d, Int offset=0 ) const;
    void GetImagPartOfDiagonal
    ( DistMatrix<BASE(T),MC,STAR>& d, Int offset=0 ) const;
    void GetRealPartOfDiagonal
    ( DistMatrix<BASE(T),STAR,MC>& d, Int offset=0 ) const;
    void GetImagPartOfDiagonal
    ( DistMatrix<BASE(T),STAR,MC>& d, Int offset=0 ) const;
    type GetDiagonal( Int offset=0 ) const;
    DistMatrix<BASE(T),MC,STAR> GetRealPartOfDiagonal( Int offset=0 ) const;
    DistMatrix<BASE(T),MC,STAR> GetImagPartOfDiagonal( Int offset=0 ) const;

    void SetDiagonal( const type& d, Int offset=0 );
    void SetDiagonal( const DistMatrix<T,STAR,MC>& d, Int offset=0 );
    void SetRealPartOfDiagonal
    ( const DistMatrix<BASE(T),MC,STAR>& d, Int offset=0 );
    // Only valid for complex datatypes
    void SetImagPartOfDiagonal
    ( const DistMatrix<BASE(T),MC,STAR>& d, Int offset=0 );
    void SetRealPartOfDiagonal
    ( const DistMatrix<BASE(T),STAR,MC>& d, Int offset=0 );
    // Only valid for complex datatypes
    void SetImagPartOfDiagonal
    ( const DistMatrix<BASE(T),STAR,MC>& d, Int offset=0 );

    void AlignWithDiagonal( const admType& A, Int offset=0 );
    void AlignWithDiagonal( const elem::DistData& data, Int offset=0 );
    bool AlignedWithDiagonal( const admType& A, Int offset=0 ) const;
    bool AlignedWithDiagonal( const elem::DistData& data, Int offset=0 ) const;

    // (Immutable) view of a distributed matrix's buffer
    void Attach
    ( Int height, Int width, Int colAlignment,
      T* buffer, Int ldim, const elem::Grid& grid );
    void LockedAttach
    ( Int height, Int width, Int colAlignment,
      const T* buffer, Int ldim, const elem::Grid& grid );

    // AllReduce sum over process row
    void SumOverRow();

private:
    template<typename S,class Function>
    void GetDiagonalHelper
    ( DistMatrix<S,MC,STAR>& d, Int offset, Function func ) const;
    template<typename S,class Function>
    void GetDiagonalHelper
    ( DistMatrix<S,STAR,MC>& d, Int offset, Function func ) const;
    template<typename S,class Function>
    void SetDiagonalHelper
    ( const DistMatrix<S,MC,STAR>& d, Int offset, Function func );
    template<typename S,class Function>
    void SetDiagonalHelper
    ( const DistMatrix<S,STAR,MC>& d, Int offset, Function func );

#ifndef SWIG
    template<typename S,Distribution U,Distribution V>
    friend class DistMatrix;
#endif // ifndef SWIG
};

} // namespace elem

#endif // ifndef ELEM_CORE_DISTMATRIX_MC_STAR_DECL_HPP
