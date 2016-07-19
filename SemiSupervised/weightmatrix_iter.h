#ifndef _WEIGHT_MATRIX_ITER
#define _WEIGHT_MATRIX_ITER

#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <cstdio>

#include "rowval.h"

#include <boost/thread.hpp>
#include <boost/ref.hpp>

// #include "GCoptimization.h"
// #include <dai/alldai.h>  // Include main libDAI header file
// #include <dai/bp.h>  // Include main libDAI header file
// #include <dai/trwbp.h>  // Include main libDAI header file

// #include "cholmod_l_solver.h"

// namespace NeuroProof{
  

class WeightMatrix_iter{

    
    std::vector< std::vector<RowVal> > _wtmat;
    std::vector< std::vector<RowVal> > _Wul;
    std::vector< std::vector<RowVal> > _Wnorm;
    
    std::set<unsigned int> _ignore;
    

    std::vector<long> _Wnorm_i;
    std::vector<long> _Wnorm_j;
    std::vector<double> _Wnorm_v;
    
    
    unsigned int _n;
 
    std::vector<double> _deltas;
    unsigned int _nrows;
    size_t _ncols;
    
    size_t _nlabeled;
    
    double _thd;
    
    std::vector<double> _degree;
    
    std::vector<int> _trn_map; //faster than multimap
    std::vector<int> _trn_lbl; 
    std::vector<int> _tst_map;

    
    size_t _trn_count;
    size_t _tst_count;
    
//     CholmodSolver cholmod_solver;
    
//     AMGsolver amgsolver;
    
    std::vector<size_t> _nzd_indicator; // nzd means non-zero degree
    std::map<unsigned int, unsigned int> _nzd_map;
    std::map<unsigned int, unsigned int> _nzd_invmap;
    size_t _nzd_count;
    size_t _nnz;
    
    std::vector<unsigned int>  _mapped_trn_idx;
    std::map<unsigned int,int>  _cum_trn_labels;
    
    std::vector<size_t> _trn_indicator; // nzd means non-zero degree
    
//     std::vector< dai::Var > vars;
//     std::vector< dai::Factor > factors;
    
    std::vector< std::vector<double> >  _prev_result;
//     std::vector< std::vector<double> >  _result;
    std::vector< std::vector<double> >  _result_parallel;
    
    int _nClass;	
    size_t _maxIter;
    
public:
  
    WeightMatrix_iter(int pclass)
    { 
	_trn_map.clear(); _trn_count = 0;
	_trn_lbl.clear(); 
	_tst_map.clear(); _tst_count = 0;
	
	
	_nzd_indicator.clear();
	_nzd_map.clear();
	_nzd_invmap.clear();
	_nzd_count = 0;
	
	_mapped_trn_idx.clear();
	_cum_trn_labels.clear();
// 	_mapped_tst_idx.clear();
	_trn_indicator.clear();
	
	_nClass=pclass;
	
    }; 
    WeightMatrix_iter(int pclass, double pthd, std::vector<unsigned int> &pignore): _thd(pthd)
    { 
	_trn_map.clear(); _trn_count = 0;
	_trn_lbl.clear(); 
	_tst_map.clear(); _tst_count = 0;
	
	for(size_t ii=0; ii < pignore.size(); ii++ )
	    _ignore.insert(pignore[ii]);
	
	_nzd_indicator.clear();
	_nzd_map.clear();
	_nzd_invmap.clear();
	_nzd_count = 0;
	
	_mapped_trn_idx.clear();
	_cum_trn_labels.clear();
// 	_mapped_tst_idx.clear();
	_trn_indicator.clear();
	
	_nClass = pclass;
	
    }; 
    ~WeightMatrix_iter(){
// 	delete gc;
// 	delete [] data;
// 	delete [] smooth;
    }
    
    void weight_matrix(std::vector< std::vector<double> >& pfeatures,
		 bool exhaustive);
    void weight_matrix_parallel(std::vector< std::vector<double> >& pfeatures,
		 bool exhaustive);

    void compute_weight_partial(std::map<double, unsigned int>& sorted_features_part,
			    std::map<double, unsigned int>& sorted_features,
			    std::vector< std::vector<double> >& pfeatures,
			    std::vector< std::vector<RowVal> >& wtmat,
			    std::vector<double>& degree);
    
    
    void EstimateBandwidth(std::vector< std::vector<double> >& pfeatures, 
			   std::vector<double>& deltas);
    double distw(std::vector<double>& vecA, std::vector<double>& vecB);
    double vec_dot(std::vector<double>& vecA, std::vector<double>& vecB);
    void scale_vec(std::vector<double>& vecA, std::vector<double>& deltas);
    double vec_norm(std::vector<double>& vecA);
    
    void find_nonzero_degree(std::vector<unsigned int>& ridx);
    void find_large_degree(std::vector<unsigned int>& ridx);
    
//     void compute_Wul();
    void compute_Wnorm();
    void solve(std::map<unsigned int, std::vector<double> >& result);
    void add2trnset(std::vector<unsigned int>& trnset, std::vector<int> &labels);
    
    void solve_partial(size_t start_row, size_t end_row);
    
    void copy(std::vector< std::vector<double> >& src, std::vector< std::vector<double> >& dst);
    void scale_features(std::vector< std::vector<double> >& allfeatures,
				    std::vector< std::vector<double> >& pfeatures);
  
    void write_matrix(char* filename);
    void read_matrix(char* filename);
    void factorize();
    void copy_factor_from(WeightMatrix_iter* sourceW);
//     CholmodSolver* get_cholmod_solver();
    
    void connection_unlabeled(unsigned int idx, int label_idx,
			    std::map< unsigned int, int >& trn_label, 
			    double& deg, std::vector<double>& cut);
    
    unsigned int num_connections(unsigned int node_idx)
    {
      return _wtmat[node_idx].size();
    };
  
    double get_degree(unsigned int node_idx)
    {
      return _degree[node_idx];
    };
    double nnz_pct();
    
    void reassign_label();
    double vector_sum(std::vector<double>& rowvec);
    void normalize_row(std::vector<double>& rowvec);
    
    
    void free_memory(){
	_Wnorm_i.clear();
	_Wnorm_j.clear();
	_Wnorm_v.clear();
// 	cholmod_solver.free_matrix();
    }
};


// }
#endif
