#include <ros/ros.h>
#include <ros/console.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/common/eigen.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "geometry_msgs/Point.h"
#include "geometry_msgs/TransformStamped.h"
#include "geometry_msgs/PointStamped.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/Image.h"
#include "pcl_segmentation_server.h"
#include "segment_cuboid.h"
#include "mask_from_cuboid.h"
#include "centroid.h"
#include "segment_bb.h"
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>

tf2_ros::Buffer * transformBuffer;
tf2_ros::TransformListener * transformListener;

int main(int argc, char** argv)
{
	ros::init(argc, argv, "pcl_segmentation_server");
	transformBuffer = new tf2_ros::Buffer(ros::Duration(30.0));
	transformListener = new tf2_ros::TransformListener(*transformBuffer);
	ros::Duration(2).sleep();
	
	ROS_INFO("Segmentation server started");

	ros::NodeHandle n;
	ros::ServiceServer segment_cuboid_service = n.advertiseService("/pcl_segmentation_server/segment_cuboid", handle_segment_cuboid);
	ros::ServiceServer segment_bb_service = n.advertiseService("/pcl_segmentation_server/segment_bb", handle_segment_bb);
	ros::ServiceServer mask_from_cuboid_service = n.advertiseService("/pcl_segmentation_server/mask_from_cuboid", handle_mask_from_cuboid);
	ros::ServiceServer centroid_service = n.advertiseService("/pcl_segmentation_server/centroid", handle_centroid);
	ros::ServiceServer segment_view_service = n.advertiseService("/pcl_segmentation_server/segment_view", handle_segment_objects);
	ros::spin();
}

void publish_pc(pcl::PointCloud<pcl::PointXYZRGB> *cloud, std::string frame) {
	// sensor_msgs::PointCloud2 output;
  	// pcl::PCLPointCloud2 outputPCL;

    // pcl::toPCLPointCloud2(*cloud ,outputPCL);
	// pcl_conversions::fromPCL(outputPCL, output);

	// output.header.frame_id = frame;
	
	// int i = 0;
	// while (i < 2) {
	// 	pub.publish(output);
	// 	ros::Duration(1.5).sleep();
	// 	i++;
	// }
	
}

bool handle_segment_objects(pcl_segmentation::SegmentView::Request& req, pcl_segmentation::SegmentView::Response& res) {
	const sensor_msgs::PointCloud2 pc = *(ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/xtion/depth_registered/points"));
	
  	pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;
  	pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);

	pcl_conversions::toPCL(pc, *cloud);

	// Perform voxel grid downsampling filtering
  	pcl::PCLPointCloud2* downSampledCloud = new pcl::PCLPointCloud2;
  	pcl::PCLPointCloud2ConstPtr downSampledCloudPtr(downSampledCloud);
	
	pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
	sor.setInputCloud(cloudPtr);
	sor.setLeafSize(0.01, 0.01, 0.01);
	sor.filter(*downSampledCloud);




	pcl::PointCloud<pcl::PointXYZRGB> *xyz_cloud = new pcl::PointCloud<pcl::PointXYZRGB>;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtr (xyz_cloud); // need a boost shared pointer for pcl function inputs

	// convert the pcl::PointCloud2 tpye to pcl::PointCloud<pcl::PointXYZRGB>
	pcl::fromPCLPointCloud2(*downSampledCloudPtr, *xyzCloudPtr);


	// create a pcl object to hold the ransac filtered results
	pcl::PointCloud<pcl::PointXYZRGB> *xyz_cloud_ransac_filtered = new pcl::PointCloud<pcl::PointXYZRGB>;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtrRansacFiltered (xyz_cloud_ransac_filtered);


	// perform ransac planar filtration to remove table top
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZRGB> seg1;
	seg1.setOptimizeCoefficients (true);
	seg1.setModelType (pcl::SACMODEL_PLANE);
	seg1.setMethodType (pcl::SAC_RANSAC);
	seg1.setDistanceThreshold (0.04);

	seg1.setInputCloud (xyzCloudPtr);
	seg1.segment (*inliers, *coefficients);


	// Create the filtering object
	pcl::ExtractIndices<pcl::PointXYZRGB> extract;

	//extract.setInputCloud (xyzCloudPtrFiltered);
	extract.setInputCloud (xyzCloudPtr);
	extract.setIndices (inliers);
	extract.setNegative (true);
	extract.filter (*xyzCloudPtrRansacFiltered);


	ros::NodeHandle n;
	ros::Publisher pub = n.advertise<sensor_msgs::PointCloud2>("PP", 5);
	sensor_msgs::PointCloud2 outputJ;
  	pcl::PCLPointCloud2 outputPCLJ;

    pcl::toPCLPointCloud2(*xyzCloudPtrRansacFiltered ,outputPCLJ);
	pcl_conversions::fromPCL(outputPCLJ, outputJ);

	outputJ.header.frame_id = pc.header.frame_id;
	
	int i = 0;
	while (i < 5) {
		pub.publish(outputJ);
		ros::Duration(1.5).sleep();
		i++;
	}


	// Create the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree->setInputCloud (xyzCloudPtrRansacFiltered);

	// create the extraction object for the clusters
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
	ec.setClusterTolerance(0.04); // 2cm
	ec.setMinClusterSize(600);
	ec.setMaxClusterSize(25000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(xyzCloudPtrRansacFiltered);
	ec.extract(cluster_indices);

	sensor_msgs::PointCloud2 output;
  	pcl::PCLPointCloud2 outputPCL;

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {

		// create a pcl object to hold the extracted cluster
		pcl::PointCloud<pcl::PointXYZRGB> *cluster = new pcl::PointCloud<pcl::PointXYZRGB>;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterPtr (cluster);

		
		// now we are in a vector of indices pertaining to a single cluster.
		// Assign each point corresponding to this cluster in xyzCloudPtrPassthroughFiltered a specific color for identification purposes
		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
			clusterPtr->points.push_back(xyzCloudPtrRansacFiltered->points[*pit]);
		

		// if (req.publish)
		// 	publish_pc(*clusterPtr, pc.header.frame_id)
		
		pcl::toPCLPointCloud2(*cluster ,outputPCL);
		pcl_conversions::fromPCL(outputPCL, output);

		output.header.frame_id = pc.header.frame_id;
	
		res.clusters.push_back(output);
  	}

	return true;
}

bool handle_segment_cuboid(pcl_segmentation::SegmentCuboid::Request& req, pcl_segmentation::SegmentCuboid::Response &res)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::fromROSMsg(req.points, *cloud);
	auto result =  segment_cuboid(cloud, req.min, req.max, cloud_out);
	pcl::toROSMsg(*cloud_out, res.points);
	res.points.header = req.points.header;
	return result;
}

bool handle_segment_bb(pcl_segmentation::SegmentBB::Request& req, pcl_segmentation::SegmentBB::Response &res)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::fromROSMsg(req.points, *cloud);
	segment_bb(cloud, req.x1, req.y1, req.x2, req.y2, cloud_out);
	pcl::toROSMsg(*cloud_out, res.points);
	res.points.header = req.points.header;
	return true;
}

bool handle_mask_from_cuboid(pcl_segmentation::MaskFromCuboid::Request& req, pcl_segmentation::MaskFromCuboid::Response &res)
{
	geometry_msgs::PointStamped min, max;
	sensor_msgs::PointCloud2::Ptr ros_cloud_tf(new sensor_msgs::PointCloud2);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tf(new pcl::PointCloud<pcl::PointXYZ>);
	min.point = req.min;
	min.header.frame_id = "map";
	min.header.stamp = req.points.header.stamp;
	max.point = req.max;
	max.header.frame_id = "map";
	max.header.stamp = req.points.header.stamp;
	
	sensor_msgs::PointCloud2 ros_cloud = req.points;

	geometry_msgs::TransformStamped transform = transformBuffer->lookupTransform(min.header.frame_id, req.points.header.frame_id, req.points.header.stamp, ros::Duration(2.0));
	tf2::doTransform(ros_cloud, *ros_cloud_tf, transform);	
	pcl::fromROSMsg(*ros_cloud_tf, *cloud_tf);

	auto result = mask_from_cuboid(cloud_tf, *ros_cloud_tf, min.point, max.point, res.mask);
	return result;
}

bool handle_centroid(pcl_segmentation::Centroid::Request& req, pcl_segmentation::Centroid::Response& res)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	pcl::fromROSMsg(req.points, *cloud);

	pcl::PointXYZRGB centroid_;

	auto result = centroid(cloud, centroid_);

	geometry_msgs::PointStamped point;
	point.header.frame_id = req.points.header.frame_id;
	point.header.stamp = req.points.header.stamp;
	point.point.x = centroid_.x;
	point.point.y = centroid_.y;
	point.point.z = centroid_.z;
	geometry_msgs::TransformStamped transform = transformBuffer->lookupTransform("map", req.points.header.frame_id, req.points.header.stamp, ros::Duration(2.0));
	tf2::doTransform(point, res.centroid, transform);
	return result;
}
