#pragma once


struct Vec3{};


struct Triangle
{
	Vec3 v[3];
};

struct DeviceMesh
{
	Vec3* vertexData;
	u32* indexData;

	// このメッシュのBVHのルートノード（BLASとして利用）
	DeviceBLASNode* bvhRoot;
};


struct DeviceInstanceData
{
	// 1. Transform (逆行列も必須)
	Matrix4x4 worldToLocal; // レイをローカル空間に変換
	Matrix4x4 localToWorld; // ヒット情報をワールド空間に戻す

	// 2. データ参照（分岐回避の鍵）
	u32 meshIndex;       // WorldData::deviceMeshes配列へのインデックス
	u32 materialIndex;   // WorldData::deviceMaterials配列へのインデックス

	// パディング（アクセス効率のため）
	u32 padding[2];
};



struct DeviceWorldData {
	// 全メッシュの配列へのポインタ
	DeviceMesh* deviceMeshes;

	// 全マテリアルの配列へのポインタ
	DeviceMaterial* deviceMaterials; // マテリアルもPOD構造体で定義

	// 全Instanceのフラットな配列へのポインタ (Group展開後)
	DeviceInstanceData* instanceDataArray;
	u32 numInstances;

	// TLAS (Top-Level Acceleration Structure) のルートノード
	// 全Instanceに対するBVH
	DeviceTLASNode* tlasRoot;
};