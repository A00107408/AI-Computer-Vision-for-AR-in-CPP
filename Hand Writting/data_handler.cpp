#include <iostream>
#include <filesystem>
#include "data_handler.h"

data_handler::data_handler()
{
	data_array = new std::vector<data *>;
	test_data = new std::vector<data *>;
	training_data = new std::vector<data *>;
	validation_data = new std::vector<data *>;
}

data_handler::~data_handler()
{
	//Free Dynamically Allocated Stuff
}

void data_handler::read_feature_vector(std::string path)
{
	uint32_t header[4]; // | Magic Num | No of Images | Row size | Col size
	unsigned char bytes[4];
	
	FILE* f = fopen(path.c_str(), "r");
	if(f)
	{
		for (int i = 0; i < 4; i++) {
			if (fread(bytes, sizeof(bytes), 1, f))
			{
				header[i] = convert_to_little_endian(bytes);
			}
		}
		printf("Done getting image file headers.\n");
		std::cout << "Getting feature vectors now, this may take a moment...patientez vous\n";
		int image_size = (header[2] * header[3]);

		int z = 0;
		for (int i = 0; i < header[1]; i++)
		{
			data *d = new data();
			uint8_t element[1];
			
			for (int j = 0; j < image_size; j++)
			{
				/*if*/ (fread(element, sizeof(element[0]), 1, f));
				//{
					d->append_to_feature_vector(element[0]);
				//}
				//else
				//{
				//	printf("Error reading feature vectors from images file.\n");
				//	exit(1);
				//}
			}
			data_array->push_back(d);
		}
		std::cout << "Successfully read and stored " << data_array->size() << " feature vectors.\n";
	}
	else
	{
		printf("Could not find feature vector file.\n");
		exit(1);
	}
}

void data_handler::read_feature_labels(std::string path)
{
	uint32_t header[2]; // | Magic Num | No. of Images 
	unsigned char bytes[4];
	
	FILE* f = fopen(path.c_str(), "r");
	if(f)
	{
		printf("Successfully opened vector labels file.\n");
		for (int i = 0; i < 2; i++) 
		{
			if (fread(bytes, sizeof(bytes), 1, f))
			{
				header[i] = convert_to_little_endian(bytes);
			}
		}

		printf("Done getting label file headers.\n");		

		for (int i = 0; i < header[1]; i++)
		{
			uint8_t element[1];
			/*if*/ (fread(element, sizeof(element[0]), 1, f));
			//{				
				data_array->at(i)->set_label(element[0]);
				
			//}
			//else
			//{
			//	printf("Error reading labels from labels file.\n");
			//	std::cout << "Reached end of labels file with " << i << "labels.\n";				
				//break;
			//	std::cin >> a;
			//	exit(1);
			//}
		}
		printf("Successfully read and stored the labels.\n");
	}
	else
	{
		printf("Error, could not find label file.\n");
		exit(1);
	}
}

void data_handler::split_data()
{
	std::cout << "Spliting the data into training and validation... patientez.\n";

	std::unordered_set<int> used_indexes;
	int train_size = data_array->size() * TRAIN_SET_PERCENT;
	int test_size = data_array->size() * TEST_SET_PERCENT;
	int valid_size = data_array->size() * VALIDATION_PERCENT;

	//TRAINING DATA
	int count = 0;
	int x = 0;
	int prevcount = 0;

	while (count < train_size)
	{
		//My laptop is not powerful enough to run the algortihm below,
		//it gets increasingly harder to find an unused feature.
		
		//int rand_index = rand() % data_array->size(); // 0 & data_array->size() -1
		//if (used_indexes.find(rand_index) == used_indexes.end())
		//{
		//	training_data->push_back(data_array->at(rand_index));
		//	used_indexes.insert(rand_index);
		//	count++;
		//}

		///The data in the file is already randomised,
		///so index 0 to 45,000 will be used to training.
		training_data->push_back(data_array->at(count));
		count++;
	}
	printf("Training data size: %lu.\n", training_data->size());

	//TEST DATA
	//count = 0; //dont bother resetting count, 
	while (count < (train_size + test_size)) //keep indexing through the data
	{
		//int rand_index = rand() % data_array->size(); // 0 & data_array->size() -1
		//if (used_indexes.find(rand_index) == used_indexes.end())
		//{
		//	test_data->push_back(data_array->at(rand_index));
		//	used_indexes.insert(rand_index);
		//	count++;
		//}
		test_data->push_back(data_array->at(count));
		count++;
	}
	printf("Test data size: %lu.\n", test_data->size());

	//VALIDATION DATA
	//count = 0;
	while (count < (train_size + test_size + valid_size) )
	{
		//int rand_index = rand() % data_array->size(); // 0 & data_array->size() -1
		//if (used_indexes.find(rand_index) == used_indexes.end())
		//{
		//	validation_data->push_back(data_array->at(rand_index));
		//	used_indexes.insert(rand_index);
		//	count++;
		//}
		validation_data->push_back(data_array->at(count));
		count++;
	}	
	printf("Validation data size: %lu.\n", validation_data->size());
}

void data_handler::count_classes()
{
	int count = 0;
	for (unsigned i = 0; i < data_array->size(); i++)
	{
		if (class_map.find(data_array->at(i)->get_label()) == class_map.end())
		{
			class_map[data_array->at(i)->get_label()] = count;
			data_array->at(i)->set_enumerated_label(count);
			count++;
		}
	}
	num_classes = count;
	printf("Successfully extracted %d unique classes.\n", num_classes);
}

uint32_t data_handler::convert_to_little_endian(const unsigned char* bytes)
{
	return (uint32_t) ( (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]) );
}

std::vector<data*>* data_handler::get_training_data()
{
	return training_data;
}

std::vector<data*>* data_handler::get_test_data()
{
	return test_data;
}

std::vector<data*>* data_handler::get_validation_data()
{
	return validation_data;
}

int main()
{
	data_handler *dh = new data_handler();
	
	dh->read_feature_vector("train-images.idx3-ubyte");
	dh->read_feature_labels("train-labels.idx1-ubyte");
	dh->split_data();
	dh->count_classes();
}

/*int main() {
	data_handler* dh = new data_handler();

	dh->read_feature_vector("train-images.idx3-ubyte");
	dh->read_feature_labels("train-labels.idx1-ubyte");
	dh->split_data();
	dh->count_classes();
}*/
