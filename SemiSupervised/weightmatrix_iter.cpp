#include "weightmatrix_iter.h"

#define EPS 0.01
#define PRED_WT 1.0
// using namespace NeuroProof;

double WeightMatrix_iter::vec_dot(std::vector<double>& vecA, std::vector<double>& vecB)
{
    double dd=0;
    size_t ll=0;
    for(size_t j = 0; j < vecA.size(); j++){
	double d1 = (vecA[j] * vecB[j]);
	ll++;
	dd += d1;
    }
    return dd;
}

void WeightMatrix_iter::compute_Wnorm()
{
    _Wnorm_i.clear(); 
    _Wnorm_j.clear(); 
    _Wnorm_v.clear();
    
    _Wnorm.clear();
    _Wnorm.resize(_nrows);
    
    for(size_t row = 0 ; row < _nrows ; row++){
      
// 	if ((! _nzd_indicator[row]) || (_trn_indicator[row]))
	if (! _nzd_indicator[row])
	    continue;
	
	unsigned int mapped_idx = _nzd_map[row];
	_Wnorm_i.push_back(mapped_idx); 
	_Wnorm_j.push_back(mapped_idx); 
// 	_Wnorm_v.push_back(_degree[row]+EPS);
	_Wnorm_v.push_back((double)(0.00001));
	
	_Wnorm[row].push_back(RowVal(row, EPS));
	
	for(size_t jj=0; jj < _wtmat[row].size(); jj++){
	    unsigned int col = _wtmat[row][jj].j;
	    double val = _wtmat[row][jj].v;
	    
// 	    if ((! _nzd_indicator[col]) || (_trn_indicator[col]))
	    if (! _nzd_indicator[col])
		continue;
	    
	    unsigned int mapped_idx2 = _nzd_map[col];
	    
	    double bias_row = (double)sqrt(_degree[row]);
	    double bias_col = (double)sqrt(_degree[col]);
	    
	    double normalized_wt = val/sqrt(_degree[row]*_degree[col]);;
	    
	    if (normalized_wt >(1+0.00001))
	      printf("normalized weight > 1\n");
	    
		
	    _Wnorm_i.push_back(mapped_idx); 
	    _Wnorm_j.push_back(mapped_idx2); 
	    _Wnorm_v.push_back((double)normalized_wt);
	    
	    _Wnorm[row].push_back(RowVal(col, normalized_wt));
		
	}
    }

    _prev_result.resize(_nrows);
    for(size_t i=0; i< _nrows; i++){
	_prev_result[i].resize(_nClass, 0.0 );
    }
  
}

void WeightMatrix_iter::reassign_label()
{
  
    size_t nLabeledSamples = _mapped_trn_idx.size();
    for(size_t i=0; i< nLabeledSamples; i++){
	unsigned int mapped_idx = _mapped_trn_idx[i];
	unsigned int idx =  _nzd_invmap[mapped_idx];
	int lbl = _cum_trn_labels[mapped_idx];
	
	_prev_result[idx].resize(_nClass,0);
	_prev_result[idx][lbl] = 1;
// 	if (lbl == 1)
// 	    _prev_result[idx][lbl] = PRED_WT;
    }    
}
double WeightMatrix_iter::vector_sum(std::vector<double>& rowvec)
{
    double sum_row=0;
    for(int cc=0; cc<_nClass; cc++){
	sum_row += rowvec[cc];
    }
    return sum_row;
}
void WeightMatrix_iter::normalize_row(std::vector<double>& rowvec)
{
    double sum_row = vector_sum(rowvec);
//     for(int cc=0; cc<_nClass; cc++){
// 	sum_row += rowvec[cc];
//     }
    if (sum_row>0){
	for(int cc=0; cc<_nClass; cc++){
	    (rowvec[cc]) /= sum_row;
	}
    }
}

void WeightMatrix_iter::solve_partial(size_t start_row, size_t end_row)
{
    for(size_t row = start_row ; row < end_row ; row++){
      
	if (! _nzd_indicator[row])
	    continue;
	
	for(size_t jj=0; jj < _Wnorm[row].size(); jj++){
	    unsigned int col = _Wnorm[row][jj].j;
	    double val = _Wnorm[row][jj].v;
	    
	    if (! _nzd_indicator[col])
		continue;
	    
	    for(int cc=0; cc<_nClass; cc++){
		(_result_parallel[row][cc]) += (val * _prev_result[col][cc]);
	    }
	}
    }
}

void WeightMatrix_iter::solve(std::map<unsigned int, std::vector<double> >& ret_result)
{
    ret_result.clear();
//     _prev_result.resize(_nrows);
//     _result.resize(_nrows);
    _result_parallel.resize(_nrows);
    for(size_t i=0; i< _nrows; i++){
// 	_prev_result[i].resize(_nClass, 0.0 );
// 	_result[i].resize(_nClass, 0.0 );
	_result_parallel[i].resize(_nClass, 0.0 );
	
    }
    printf("LabelProp labeled example %u\n", _mapped_trn_idx.size());
    reassign_label();
    
    size_t ncores=8;
    _maxIter=200;
    double max_diff;

    
    for (size_t iter=0; iter< _maxIter; iter++){
	max_diff = 0;
      
// 	for(size_t row = 0 ; row < _nrows ; row++){
// 	  
//     // 	if ((! _nzd_indicator[row]) || (_trn_indicator[row]))
// 	    if (! _nzd_indicator[row])
// 		continue;
// 	    
// 	    for(size_t jj=0; jj < _Wnorm[row].size(); jj++){
// 		unsigned int col = _Wnorm[row][jj].j;
// 		double val = _Wnorm[row][jj].v;
// 		
//     // 	    if ((! _nzd_indicator[col]) || (_trn_indicator[col]))
// 		if (! _nzd_indicator[col])
// 		    continue;
// 		
// 		for(int cc=0; cc<_nClass; cc++){
// 		    (_result[row][cc]) += (val * _prev_result[col][cc]);
// 		}
// 	    }
// 	}
	
	size_t start_row=0;
	std::vector< boost::thread* > threads;
	size_t chunksz = _nrows/ncores;
	if (_nrows> (ncores*chunksz))
	    ncores++;
	for(size_t ichunk=0; ichunk < ncores; ichunk++){
	    size_t end_row = start_row + chunksz;
	    if (end_row >=_nrows)
		end_row = _nrows;
	    
	    threads.push_back(new boost::thread(&WeightMatrix_iter::solve_partial, this, start_row, end_row));
	    
	    start_row = end_row;
	}
	
// 	printf("Sync all threads \n");
	for (size_t ti=0; ti<threads.size(); ti++) 
	  (threads[ti])->join();
// 	printf("all threads done\n");
	
// 	double debug_diff=0;
// 	for(size_t row = 0 ; row < _nrows ; row++){
// 	    for(int cc=0; cc<_nClass; cc++){
// 		debug_diff += fabs(_result[row][cc] - _result_parallel[row][cc]);
// 	    }
// 	}
// 	if (debug_diff>0)
// 	    printf("Difference between serial and parallel = %f\n",debug_diff);
	
	
	
	
	
	for(size_t row = 0 ; row < _nrows ; row++){
// 	    normalize_row(_result[row]);
	    normalize_row(_result_parallel[row]);
	}
	
	
	for(size_t row = 0 ; row < _nrows ; row++){
	    if (_trn_indicator[row]){
	      continue;
	    }
	    
	    double diff=0;
	    for(int cc=0; cc<_nClass; cc++){
		diff = fabs(_prev_result[row][cc] - _result_parallel[row][cc]);
		max_diff = (max_diff>diff) ? max_diff: diff;
		
		_prev_result[row][cc] = _result_parallel[row][cc];
	    }
	}
	
	reassign_label();
	if (!(iter%10))
	  printf("max diff %.5f\n",max_diff);    
	if (max_diff<EPS)
	  break;
	
    }
    printf("max diff %.5f\n",max_diff);    
    
    for(size_t row = 0 ; row < _nrows ; row++){
	  
	if ((! _nzd_indicator[row]) || (_trn_indicator[row]))
	  continue;
	
	  
	
	if (!(vector_sum(_result_parallel[row])>0))
	    _result_parallel[row].assign(_nClass, (double)(1.0/_nClass));

	ret_result.insert(std::make_pair(row, _result_parallel[row]));
    }
    
}



void WeightMatrix_iter::add2trnset(std::vector<unsigned int>& trnset, std::vector<int> &trnlabels)
{
    std::vector<unsigned int> mapped_new_trn_idx;
    std::vector<int> new_trn_labels;
    
    /* debug *
    FILE* fp=fopen("tmp_check_map.txt", "wt");
    std::map<unsigned int, unsigned int>::iterator nzdit = _nzd_map.begin();
    for(; nzdit != _nzd_map.end(); nzdit++)
	fprintf(fp,"%u %u\n", nzdit->first, nzdit->second);
    fclose(fp);
    //**/
    
    for(size_t ii=0; ii < trnset.size(); ii++){
	unsigned int idx1 = trnset[ii]; 
	if (_nzd_indicator[idx1]){
	    unsigned int mapped_idx = _nzd_map[idx1];
	    _mapped_trn_idx.push_back(mapped_idx);
	    int lbl01 = trnlabels[ii] < 0 ? 0 : 1;
	    _cum_trn_labels.insert(std::make_pair(mapped_idx,lbl01));
	    
	    _trn_indicator[idx1] = 1;
	    
	    mapped_new_trn_idx.push_back(mapped_idx);
	    new_trn_labels.push_back(lbl01);
	}
    }
}


void WeightMatrix_iter::connection_unlabeled(unsigned int idx, int label_idx,
			    std::map< unsigned int, int >& trn_label, 
			    double& deg, std::vector<double>& cut)
{
    std::vector<RowVal> &row1 = _wtmat[idx];
    double wt_labeled = 0;
    double wt_other = 0;
    
    deg = 0;
    //_degree[idx];
//     if(deg == 0)
//       return;
      
    for(size_t col=0; col< row1.size(); col++){
	unsigned int nbr_idx = row1[col].j;
	double nbr_wt = row1[col].v;
	int label1 = -1;
	if (trn_label.find(nbr_idx) != trn_label.end()){ //labeled example
	    label1 = trn_label[nbr_idx];
	    
	}
	if (label1 == label_idx){
	    wt_labeled += nbr_wt;
	    (cut[label1]) += nbr_wt;
	    deg += nbr_wt;
	}
	else if (label1 >= 0){
	    wt_other += nbr_wt;
	    (cut[label1]) += nbr_wt;
	    deg += nbr_wt;
	}
	
    }
//     cut = wt_other;
    double wt_other1 = deg - wt_labeled;
    
}




//*************************************************************************************************
void WeightMatrix_iter::copy(std::vector< std::vector<double> >& src, std::vector< std::vector<double> >& dst)
{
    size_t nrows = src.size();
    size_t ncols = src[0].size() - _ignore.size();
    
    dst.clear();
    dst.resize(nrows);
    for(size_t rr=0; rr < nrows; rr++){
	dst[rr].resize(ncols);
	for(size_t cc=0, cc1=0; cc < src[rr].size(); cc++){
	    if ( (cc1 < ncols) && (_ignore.find(cc) == _ignore.end()))
	      dst[rr][cc1++] = src[rr][cc]; 
	}
    }
    
}
void WeightMatrix_iter::EstimateBandwidth(std::vector< std::vector<double> >& pfeatures,
				      std::vector<double>& deltas)
{
    deltas.clear();
    deltas.resize(_ncols);
    for(size_t col=0; col< _ncols; col++){
	// *C* compute mean
	double mean= 0;
	for(size_t row = 0; row < _nrows; row++){
	    mean += pfeatures[row][col];
	}
	mean /= _nrows;
	
	// *C* compute std dev
	double sqdev = 0;
	for(size_t row=0; row < _nrows; row++){
	    sqdev += ((pfeatures[row][col] -mean)*(pfeatures[row][col] -mean));
	}
	sqdev /= _nrows;
	
	deltas[col] = sqrt(sqdev);
    }
  
}

double WeightMatrix_iter::distw(std::vector<double>& vecA, std::vector<double>& vecB)
{
    double dd=0;
    size_t ll=0;
    for(size_t j = 0; j < vecA.size(); j++){
	double d1 = (vecA[j] - vecB[j])*(vecA[j] - vecB[j]);
	double d2 = d1 / (_deltas[j]*_deltas[j]);
	ll++;
	
	dd += d2;
    }
    //dd /= ll;
    return dd;
}
// double WeightMatrix_iter::vec_dot(std::vector<double>& vecA, std::vector<double>& vecB)
// {
//     double dd=0;
//     size_t ll=0;
//     for(size_t j = 0; j < vecA.size(); j++){
// 	double d1 = (vecA[j] * vecB[j]);
// 	ll++;
// 	dd += d1;
//     }
//     return dd;
// }

void WeightMatrix_iter::scale_vec(std::vector<double>& vecA, std::vector<double>& deltas)
{
  
    for(size_t j=0; j< vecA.size(); j++){
	vecA[j] /= deltas[j];
    }
       
}

double WeightMatrix_iter::vec_norm(std::vector<double>& vecA)
{
    double nrm = 0;
    for(size_t j=0; j< vecA.size(); j++){
	nrm += (vecA[j]*vecA[j]);
    }
    return nrm;   
}

void WeightMatrix_iter::find_nonzero_degree(std::vector<unsigned int>& ridx)
{
  
    ridx.clear();
    
    std::vector<double> degree(_nrows,0);
    
    for(size_t rr = 0; rr < _wtmat.size(); rr++){
	if (_degree[rr]>0){
	    ridx.push_back(rr);
	}
    }
}
void WeightMatrix_iter::find_large_degree(std::vector<unsigned int>& ridx){
  
    std::multimap<double, unsigned int> sorted_degree;
    for(size_t ii=0; ii < _nrows ; ii++){
	sorted_degree.insert(std::make_pair(_degree[ii], ii));
    }
    
    std::vector<unsigned int> flag(_nrows, 0);
    std::multimap<double, unsigned int>::reverse_iterator sit = sorted_degree.rbegin();
    for(; sit!= sorted_degree.rend(); sit++){
	unsigned int sidx = sit->second;
	if (!flag[sidx]){
	    flag[sidx] = 1;
	    ridx.push_back(sidx);
	    std::vector<RowVal>& nbr_wts = _wtmat[sidx];
	    for(size_t ii=0; ii < nbr_wts.size(); ii++ ){
		unsigned int sjdx = nbr_wts[ii].j;
		flag[sjdx] = 2;
	    }
	}
	  
    }
    
}

void WeightMatrix_iter::scale_features(std::vector< std::vector<double> >& allfeatures,
				    std::vector< std::vector<double> >& pfeatures)
{
    copy(allfeatures, pfeatures);


    std::vector<double> deltas;
    EstimateBandwidth(pfeatures, deltas);
    for(size_t i=0; i < pfeatures.size(); i++){
	std::vector<double>& tmpvec = pfeatures[i];
	scale_vec(tmpvec, deltas);
    }
}

void WeightMatrix_iter::weight_matrix(std::vector< std::vector<double> >& allfeatures,
			   bool exhaustive)
{

  
    std::vector< std::vector<double> > pfeatures;
    copy(allfeatures, pfeatures);

    _nrows = pfeatures.size();
    _ncols = pfeatures[0].size();

    EstimateBandwidth(pfeatures, _deltas);
    
    _wtmat.resize(_nrows);
    _degree.resize(_nrows, 0.0);
    
    unsigned long nnz=0;
   
    std::map<double, unsigned int> sorted_features;
    for(size_t i=0; i < pfeatures.size(); i++){
	std::vector<double>& tmpvec = pfeatures[i];
	scale_vec(tmpvec, _deltas);
	double nrm = vec_norm(tmpvec);
	sorted_features.insert(std::make_pair(nrm, i));
    }
    std::map<double, unsigned int>::iterator sit_i;    
    std::map<double, unsigned int>::iterator sit_j;    
    
    
    for(sit_i = sorted_features.begin() ; sit_i != sorted_features.end(); sit_i++){
	
	sit_j = sit_i;
	sit_j++;
	size_t i = sit_i->second;
	unsigned int nchecks = 0;
	for(; sit_j != sorted_features.end(); sit_j++){
	    size_t j = sit_j->second;
	    double dot_ij = vec_dot(pfeatures[i], pfeatures[j]);  
	    double dist = sit_i->first + sit_j->first - 2*dot_ij;
	    double min_dist = sit_i->first + sit_j->first - 2*sqrt(sit_i->first)*sqrt(sit_j->first);
	    
	    nchecks++;
	    
	    if (dist < (_thd*_thd)){
		double val =  exp(-dist/(0.5*_thd*_thd));
		
		_wtmat[i].push_back(RowVal(j, val));
		(_degree[i]) += val;
		_wtmat[j].push_back(RowVal(i, val));
		(_degree[j]) += val;
		
		nnz += 2;
	    }
	    if (min_dist>(_thd*_thd)){
	      break;
	    }
	}
    }
    printf("total nonzeros: %lu\n",nnz);

    /* C debug*
    FILE* fp=fopen("semi-supervised/tmp_wtmatrix.txt", "wt");
    for(size_t rr = 0; rr < _wtmat.size(); rr++){
	double luu = 0;
	double w_l = 0;
	for(size_t cc=0; cc < _wtmat[rr].size(); cc++){
	    size_t cidx = _wtmat[rr][cc].j;
	    double val = _wtmat[rr][cc].v;
	    //val = exp(-val/(0.5*_thd*_thd));
	    fprintf(fp, "%u %u %lf\n", rr, cidx, val);
	}
    }
    fclose(fp);
    //*C*/
    
    
}

void WeightMatrix_iter::weight_matrix_parallel(std::vector< std::vector<double> >& allfeatures,
			   bool exhaustive)
{

  
    printf("running parallel version\n");
    std::vector< std::vector<double> > pfeatures;
    copy(allfeatures, pfeatures);

    _nrows = pfeatures.size();
    _ncols = pfeatures[0].size();

    EstimateBandwidth(pfeatures, _deltas);
    
    _wtmat.resize(_nrows);
    _degree.resize(_nrows, 0.0);
    
    size_t NCORES = 8;
   
    std::map<double, unsigned int> sorted_features; // all features are unique, check learn or iterative learn algo
    std::vector< std::map<double, unsigned int> > sorted_features_p(NCORES); // all features are unique, check learn or iterative learn algo
    int part = 0;  
    for(size_t i=0; i < pfeatures.size(); i++){
	std::vector<double>& tmpvec = pfeatures[i];
	scale_vec(tmpvec, _deltas);
	double nrm = vec_norm(tmpvec);
	sorted_features.insert(std::make_pair(nrm, i));
	
	(sorted_features_p[part]).insert(std::make_pair(nrm, i));
	part = (part+1) % NCORES;
    }
    
    std::vector< boost::thread* > threads;
    std::vector< std::vector< std::vector<RowVal> > > tmpmat(sorted_features_p.size());
    std::vector< std::vector<double> > degrees(sorted_features_p.size());
    for(size_t pp=0; pp < sorted_features_p.size(); pp++){
      tmpmat[pp].resize(_nrows);
      degrees[pp].resize(_nrows, 0.0);
      //compute_weight_partial(sorted_features_p[i], sorted_features, pfeatures);
      threads.push_back(new boost::thread(&WeightMatrix_iter::compute_weight_partial, this, sorted_features_p[pp], sorted_features, pfeatures, boost::ref(tmpmat[pp]), boost::ref(degrees[pp])));
    }
    
    printf("Sync all threads \n");
    for (size_t ti=0; ti<threads.size(); ti++) 
      (threads[ti])->join();
    printf("all threads done\n");
    
    for(size_t pp=0; pp < tmpmat.size(); pp++){
	for(size_t i=0; i < tmpmat[pp].size(); i++){
	    if(tmpmat[pp][i].size()>0){
	       _wtmat[i].insert(_wtmat[i].end(),tmpmat[pp][i].begin(), tmpmat[pp][i].end());
	       (_degree[i]) += degrees[pp][i];
	    }
	}
    }    
    
    if(_nrows != _wtmat.size()){
	printf(" number of rows and wtmat size mismatch\n");
    }
    _nzd_indicator.resize(_nrows, 0);
    _trn_indicator.resize(_nrows, 0);
    for (size_t i=0; i < _nrows; i++){
	
	if (_degree[i] > 0){
	    _nzd_indicator[i] = 1;
	    _nzd_map.insert(std::make_pair(i, _nzd_count) );
	    _nzd_invmap.insert(std::make_pair(_nzd_count, i) );
	    _nzd_count++;
	}
	  
    }
    
    compute_Wnorm();
//     //** _nzd_count = non-zero degree count == number of columns 
//     //** _nnz = number of nonzeros
//     
//     _nnz = _Wnorm_i.size();
//     cholmod_solver.initialize(_nzd_count, _nnz, _Wnorm_i.data(), _Wnorm_j.data(), _Wnorm_v.data());
    
    /* C debug*
    FILE* fp=fopen("semi-supervised/tmp_wtmatrix_parallel.txt", "wt");
    for(size_t rr = 0; rr < _wtmat.size(); rr++){
	double luu = 0;
	double w_l = 0;
	for(size_t cc=0; cc < _wtmat[rr].size(); cc++){
	    size_t cidx = _wtmat[rr][cc].j;
	    double val = _wtmat[rr][cc].v;
	    //val = exp(-val/(0.5*_thd*_thd));
	    fprintf(fp, "%u %u %lf\n", rr, cidx, val);
	}
    }
    fclose(fp);
    //*C*/
    
    
}
void WeightMatrix_iter::write_matrix(char* filename)
{
  
    FILE* fp=fopen(filename, "wt");
    fprintf(fp,"%u\n",_nrows);
    for(size_t rr = 0; rr < _wtmat.size(); rr++){
	double luu = 0;
	double w_l = 0;
	for(size_t cc=0; cc < _wtmat[rr].size(); cc++){
	    size_t cidx = _wtmat[rr][cc].j;
	    double val = _wtmat[rr][cc].v;
	    //val = exp(-val/(0.5*_thd*_thd));
	    fprintf(fp, "%u %u %lf\n", rr, cidx, val);
	}
    }
    fclose(fp);
}

void WeightMatrix_iter::read_matrix(char* filename)
{
    FILE* fp=fopen(filename, "rt");
    fscanf(fp,"%u\n",&_nrows);
    _wtmat.clear();_degree.clear();
    _wtmat.resize(_nrows);
    _degree.resize(_nrows,0);
    unsigned int i,j;
    double val;
    while(!feof(fp)){
	fscanf(fp, "%u %u %f\n", &i, &j, &val);
	_wtmat[i].push_back(RowVal(j, val));
	(_degree[i]) += val;
    }
    fclose(fp);
    if(_nrows != _wtmat.size()){
	printf(" number of rows and wtmat size mismatch\n");
    }
    _nzd_indicator.resize(_nrows, 0);
    _trn_indicator.resize(_nrows, 0);
    for (size_t i=0; i < _nrows; i++){
	
	if (_degree[i] > 0){
	    _nzd_indicator[i] = 1;
	    _nzd_map.insert(std::make_pair(i, _nzd_count) );
	    _nzd_invmap.insert(std::make_pair(_nzd_count, i) );
	    _nzd_count++;
	}
	  
    }
    
    compute_Wnorm();
    
    
//     //** _nzd_count = non-zero degree count == number of columns 
//     //** _nnz = number of nonzeros
//     
//     _nnz = _Wnorm_i.size();
//     cholmod_solver.initialize(_nzd_count, _nnz, _Wnorm_i.data(), _Wnorm_j.data(), _Wnorm_v.data());
//     
    
}
void WeightMatrix_iter::factorize()
{
//     cholmod_solver.factorize();
}

void WeightMatrix_iter::copy_factor_from(WeightMatrix_iter* sourceW)
{
//     cholmod_solver.copy_factor_from(sourceW->get_cholmod_solver());
}
// CholmodSolver* WeightMatrix_iter::get_cholmod_solver()
// {
//     return &cholmod_solver;
// }

void WeightMatrix_iter::compute_weight_partial(std::map<double, unsigned int>& sorted_features_part,
			    std::map<double, unsigned int>& sorted_features,
			    std::vector< std::vector<double> >& pfeatures,
			    std::vector< std::vector<RowVal> >& wtmat,
			    std::vector<double>& degree)
{
  
    unsigned long nnz=0;
    std::map<double, unsigned int>::iterator sit_i;    
    std::map<double, unsigned int>::iterator sit_j;    
    for(sit_i = sorted_features_part.begin() ; sit_i != sorted_features_part.end(); sit_i++){
	
	size_t i = sit_i->second;
	double norm_i = sit_i->first;
	sit_j = sorted_features.find(norm_i);
	sit_j++;
	unsigned int nchecks = 0;
	for(; sit_j != sorted_features.end(); sit_j++){
	    size_t j = sit_j->second;
	    double dot_ij = vec_dot(pfeatures[i], pfeatures[j]);  
	    double dist = sit_i->first + sit_j->first - 2*dot_ij;
	    double min_dist = sit_i->first + sit_j->first - 2*sqrt(sit_i->first)*sqrt(sit_j->first);
	    
	    nchecks++;
	    
	    if (dist < (_thd*_thd)){
		double val =  exp(-dist/(0.5*_thd*_thd));
		
		wtmat[i].push_back(RowVal(j, val));
		(degree[i]) += val;
		wtmat[j].push_back(RowVal(i, val));
		(degree[j]) += val;
		
		nnz += 1;
	    }
	    if (min_dist>(_thd*_thd)){
	      break;
	    }
	}
    }
    printf("total nonzeros: %lu\n",nnz);

}
double WeightMatrix_iter::nnz_pct()
{
      
    double pct = 0.0;
    size_t nr = _degree.size();
    for(size_t ii=0; ii < nr ; ii++)
	pct += _wtmat[ii].size();
    
    printf("Wt Mat nnz: %u, total:%u\n",(unsigned int)pct, nr);
    pct /= (nr*nr);
    printf("Wt Mat nnz pct: %lf\n", pct);
    return pct;
}


