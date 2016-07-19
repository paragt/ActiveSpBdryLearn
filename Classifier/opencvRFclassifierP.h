
#ifndef _opencv_rf_classifierP
#define _opencv_rf_classifierP

#include "opencv/ml.h"
#include <boost/thread.hpp>

#include "edgeclassifier.h"

using namespace std;


class OpencvRFclassifierP: public EdgeClassifier{

    CvRTrees* _rf;	
    int _nfeatures;
    int _nclass;	

    vector<CvForestTree*> _trees;

    vector<double> _tree_weights;	

    bool _use_tree_weights; 

    int _tree_count;
    int _max_depth;	
		
    std::vector<unsigned int> ignore_featlist;
    size_t _ncores;
    std::vector< CvRTrees* > _rfvec;

public:
     OpencvRFclassifierP():_rf(NULL), _tree_count(256), _max_depth(20), _ncores(8) {};	
     OpencvRFclassifierP(int ptree_count, int pmax_depth):_rf(NULL), _tree_count(ptree_count), _max_depth(pmax_depth), _ncores(8) {};	
     OpencvRFclassifierP(const char* rf_filename);
     ~OpencvRFclassifierP(){
	 if (_rf) delete _rf;
     }	
     void  load_classifier(const char* rf_filename);
     double predict(std::vector<double>& features);
     void learn(std::vector< std::vector<double> >& pfeatures, std::vector<int>& plabels);
     void save_classifier(const char* rf_filename);

     void set_tree_weights(vector<double>& pwts);	
     void get_tree_responses(vector<double>& pfeatures,vector<double>& responses);	
     void reduce_trees();	

     void set_ignore_featlist(std::vector<unsigned int>& pignore_list){ignore_featlist = pignore_list;};
     void get_ignore_featlist(std::vector<unsigned int>& pignore_list){pignore_list = ignore_featlist;};
     
     void learn_part(size_t icore, CvMat *features, CvMat *labels);
     

     bool is_trained(){
	if (_rf && _rf->get_tree_count()>0)
	   return true;
	else 
	   return false;
		
     };

};

#endif
