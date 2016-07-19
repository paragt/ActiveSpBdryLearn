#include "vigraRFclassifierP.h"
#include "assert.h"
// #include <time.h>
#include <ctime>

VigraRFclassifierP::VigraRFclassifierP(const char* rf_filename){

    ncores = 8;
    load_classifier(rf_filename);
}

void  VigraRFclassifierP::load_classifier(const char* rf_filename){
    _rfvec.clear();
    _rfvec.resize(ncores);
    
    HDF5File rf_file(rf_filename,HDF5File::Open); 	
     string rf_dataset="rf";
     char buffer[256];
     for(size_t icore=0; icore<ncores; icore++){
	sprintf(buffer,"%u",icore);
	string rf_dataseti = rf_dataset +"_"+ buffer;
	rf_import_HDF5(*(_rfvec[icore]), rf_file,rf_dataseti.c_str());
     }
//     rf_import_HDF5(*_rf, rf_file,"rf");
    printf("RF loaded with %d trees, for %d class prob with %d dimensions\n",ncores*_rfvec[0]->tree_count(), _rfvec[0]->class_count(), _rfvec[0]->column_count());
    _nfeatures = _rfvec[0]->column_count();
    _nclass = _rfvec[0]->class_count();
     
    /* read list of useless features*/ 
    string filename = rf_filename;
    unsigned found = filename.find_last_of(".");
    string nameonly = filename.substr(0,found);	
    nameonly += "_ignore.txt";
    FILE* fp = fopen(nameonly.c_str(),"rt");
    if (!fp){
	printf("no features to ignore\n");
	return;
    }
    else{
	size_t i=0;
	unsigned int a;
	while(!feof(fp)){
	    fscanf(fp,"%u ", &a);
	    ignore_featlist.push_back(a);
	    i++;
	}
	fclose(fp);
    }
}

double VigraRFclassifierP::predict(std::vector<double>& features){
  
    if(_rfvec.size()<1){
	printf("no trees learned\n");
	exit(0);
    }
    assert(_nfeatures == features.size());
    MultiArray<2, float> vfeatures(Shape(1,_nfeatures));
    MultiArray<2, float> prob(Shape(1, _nclass));
    for(int i=0;i<_nfeatures;i++)
	  vfeatures(0,i)= (float)features[i];

    size_t indiv_tree_count = total_tree_count / ncores;

     std::vector<double> vp(_nclass,0.0);
     std::vector<double> vp_tmp(_nclass,0.0);
     for(size_t icore=0; icore<ncores; icore++){
	_rfvec[icore]->predictProbabilities(vfeatures, prob);    
	for(int i=0;i<_nclass;i++){
	    vp_tmp[i] = prob(0,i);
	    (vp[i]) += (prob(0,i));	
	}
     }

    /*debug*/
//     for(int i=0;i<_nclass;i++)
// 	vp[i] = prob(0,i);	
    /**/

    return (double) (vp[1]*1.0/ncores);
			    
}	


void VigraRFclassifierP::learn(std::vector< std::vector<double> >& pfeatures, std::vector<int>& plabels){

//      if (_rf)
// 	delete _rf;	
     

     int rows = pfeatures.size();
     int cols = pfeatures[0].size();	 	
     
     printf("Number of samples and dimensions: %d, %d\n",rows, cols);
     if ((rows<1)||(cols<1)){
	return;
     }

//      clock_t start = clock();
     std::time_t start, end;
//      std::time(&start);	

     MultiArray<2, float> features(Shape(rows,cols));
     MultiArray<2, int> labels(Shape(rows,1));

     int numzeros=0; 	
     for(int i=0; i < rows; i++){
	 labels(i,0) = plabels[i];
	 numzeros += (labels(i,0)==-1?1:0);
	 for(int j=0; j < cols ; j++){
	     features(i,j) = (float)pfeatures[i][j];	
	 }
     }
     printf("Number of merge: %d\n",numzeros);


    total_tree_count = 256;
    ncores=8;
    
     printf("Number of trees:  %d\n",total_tree_count);
//      RandomForestOptions rfoptions = RandomForestOptions().tree_count(tre_count).use_stratification(RF_EQUAL).min_split_node_size(10);	//RF_EQUAL, RF_PROPORTIONAL
//      _rf= new RandomForest<>(rfoptions);


     // construct visitor to calculate out-of-bag error
//      visitors::OOB_Error oob_v;
//      visitors::VariableImportanceVisitor varimp_v;
// 
//      _rf->learn(features, labels);
//      _nfeatures = _rf->column_count();
//      _nclass = _rf->class_count();

//      std::time(&end);
//      printf("Time required to learn RF: %.2f sec\n", (difftime(end,start))*1.0);

     std::time(&start);	
     _rfvec.clear();
     _rfvec.resize(ncores);
     std::vector< boost::thread* > threads;
     for(size_t icore=0; icore<ncores; icore++){
	threads.push_back(new boost::thread(&VigraRFclassifierP::learn_part,this,icore,features,labels));	
     }
     for (size_t ti=0; ti<threads.size(); ti++) 
      (threads[ti])->join();
     printf("all threads done\n");
     
     std::time(&end);
     printf("Time required to learn RF parallel: %.2f sec\n", (difftime(end,start))*1.0);
     
     _nfeatures = _rfvec[0]->column_count();
     _nclass = _rfvec[0]->class_count();
    printf("parallel vigra class count: %u\n",_nclass);
//      printf("Time required to learn RF: %.2f\n", ((double)clock() - start) / CLOCKS_PER_SEC);
//      printf("with oob :%f\n", oob_v.oob_breiman);
     
     
}
void VigraRFclassifierP::learn_part(size_t tree_idx, MultiArray<2, float>& vfeatures, MultiArray<2, int>& vlabels)
{
    size_t indiv_tree_count = total_tree_count / ncores;
  
    RandomForestOptions rfoptionsp = RandomForestOptions().tree_count(indiv_tree_count).use_stratification(RF_EQUAL).min_split_node_size(10);	//RF_EQUAL, RF_PROPORTIONAL
    _rfvec[tree_idx] = new RandomForest<>(rfoptionsp);
    _rfvec[tree_idx]->learn(vfeatures, vlabels);
        
}
void VigraRFclassifierP::save_classifier(const char* rf_filename){
  
    if (ignore_featlist.size()>0){
	string filename = rf_filename;
	unsigned found = filename.find_last_of(".");
	string nameonly = filename.substr(0,found);	
	nameonly += "_ignore.txt";
	FILE* fp = fopen(nameonly.c_str(),"wt");
	for(size_t i=0; i<ignore_featlist.size(); i++){
	    fprintf(fp,"%u ", ignore_featlist[i]);
	}
	fclose(fp);
    }
    
     HDF5File rf_file(rf_filename,HDF5File::New); 	
     string rf_dataset="rf";
     char buffer[256];
     for(size_t icore=0; icore<ncores; icore++){
       sprintf(buffer, "%u",icore);
	string rf_dataseti = rf_dataset +"_"+ buffer;
	rf_export_HDF5(*(_rfvec[icore]), rf_file,rf_dataseti.c_str());
     }
}

