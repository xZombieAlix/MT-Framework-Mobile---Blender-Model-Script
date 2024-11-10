# Copyright 2015 Seth VanHeulen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

bl_info= {
    "name": "Imports/Exports Mercenaries 3D Models",
    "author": "Converted from MH4U Importer from Seth VanHeulen",
    "version": (1, 1),
    "blender": (3, 6, 0),
    "location": "File > Import > Mercenaries 3D Model (.mod)",
    "description": "Imports/Exports a Mercenaries 3D model.",
    "category": "Import-Export",
}
# Credits - code snippets adapted from following projects:
#   Seth VanHeulen : Monster Hunter 4 Ultimate model importer
#   HZDMeshTool - AlexPo: Skelton/armature creation , vertex/UV/norml exporting code
#   Ablam - HenryOfCarim and Sebastian Brachi : tri-list to tri-strip code
# 
# 11/3: 1.0, Aman,Initial version
# 11/8: 1.1, Aman, added a zero weight bone for export vertices. Should set Bone Total limits to 2 for weight transfer.   

import array
import struct

import bpy
import bmesh
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty, FloatProperty, CollectionProperty
import operator

import mathutils
import re
from collections import deque, namedtuple

modifier_tables = (
    (2, 8, -2, -8),
    (5, 17, -5, -17),
    (9, 29, -9, -29),
    (13, 42, -13, -42),
    (18, 60, -18, -60),
    (24, 80, -24, -80),
    (33, 106, -33, -106),
    (47, 183, -47, -183)
)

# code from Ablam Reloaded Blender addon (HenryOfCarim and Sebastian Brachi) 
def triangles_list_to_triangles_strip(mesh):
    """
    Export triangle strips from a blender mesh.
    It assumes the mesh is all triangulated.
    Based on a paper by Pierre Terdiman: http://www.codercorner.com/Strips.htm
    """
    # TODO: Fix changing of face orientation in some cases (see tests)
    edges_faces = {}
    current_strip = []
    strips = []
    joined_strips = []
    #face_cnt = int(len(mesh.indices)/3)
    face_cnt = len(mesh.polygons)
    faces_indices = deque(p for p in range(face_cnt))
    done_faces_indices = set()
    current_face_index = faces_indices.popleft()
    process_faces = True
    face_edges = []
    face_verts = []
    
    # build per face vertex array  and edge edge from tri list
    #tri_verts = deque(v for v in mesh.indices)
    tri_verts = deque()
    for poly in mesh.polygons:
        for v in poly.vertices:
            tri_verts.append(v)

    
    while tri_verts:
        v1=tri_verts.popleft()
        v2=tri_verts.popleft()
        v3=tri_verts.popleft()
        if v2 > v1: 
            e1 = (v1,v2) 
        else: 
            e1 = (v2,v1)
        if v3 > v2: 
            e2 = (v2,v3) 
        else: 
            e2 = (v3,v2)
        if v3 > v1: 
            e3 = (v1,v3) 
        else: 
            e3 = (v3,v1)        
        face_edges.append([e1,e2,e3])        
        face_verts.append([v1,v2,v3])
    # edges_faces collect faces that share the same edge    
    for index, edges in enumerate(face_edges):
        for edge in edges:
            edges_faces.setdefault(edge, set()).add(index)

    while process_faces:
        current_face_verts = face_verts[current_face_index][:] 
        strip_indices = [v for v in current_face_verts if v not in current_strip[-2:]]
        if current_strip:
            face_to_add = tuple(current_strip[-2:]) + tuple(strip_indices)
            if face_to_add != current_face_verts and face_to_add != tuple(reversed(current_face_verts)):
                # we arrived here because the current face shares and edge with the face in the strip
                # however, if we just add the verts, we would be changing the direction of the face
                # so we create a degenerate triangle before adding to it to the strip
                current_strip.append(current_strip[-2])
        current_strip.extend(strip_indices)
        done_faces_indices.add(current_face_index)

        next_face_index = None
        possible_face_indices = {}
        for edge in face_edges[current_face_index]:
            if edge not in edges_faces:
                continue
            checked_edge = {face_index: edge for face_index in edges_faces[edge]
                            if face_index != current_face_index and face_index not in done_faces_indices}
            possible_face_indices.update(checked_edge)
        for face_index, edge in possible_face_indices.items():
            if not current_strip:
                next_face_index = face_index
                break
            elif edge == tuple(current_strip[-2:]) or edge == tuple(reversed(current_strip[-2:])):
                next_face_index = face_index
                break
            elif edge == (current_strip[-1], current_strip[-2]):
                if len(current_strip) % 2 != 0:
                    # create a degenerate triangle to join them
                    current_strip.append(current_strip[-2])
                next_face_index = face_index

        if next_face_index:
            faces_indices.remove(next_face_index)
            current_face_index = next_face_index
        else:
            strips.append(current_strip)
            current_strip = []
            try:
                current_face_index = faces_indices.popleft()
            except IndexError:
                process_faces = False

    prev_strip_len = 0
    # join strips with degenerate triangles
    for strip in strips:
        if not prev_strip_len:
            joined_strips.extend(strip)
            prev_strip_len = len(strip)
        elif prev_strip_len % 2 == 0:
            joined_strips.extend((joined_strips[-1], strip[0]))
            joined_strips.extend(strip)
            prev_strip_len = len(strip)
        else:
            joined_strips.extend((joined_strips[-1], strip[0], strip[0]))
            joined_strips.extend(strip)
            prev_strip_len = len(strip)
           
    # make sure joined_strip len is multiple of 4
    strip_len = len(joined_strips)
    padding = int((strip_len+3)/4)*4 - strip_len
    last_idx = joined_strips[-1]
    for n in range(padding):
        joined_strips.append(last_idx)
    return joined_strips


def decode_etc1(image, data):
    data = array.array('I', data)
    image_pixels = [0.0, 0.0, 0.0, 1.0] * image.size[0] * image.size[1]
    block_index = 0
    while len(data) != 0:
        alpha_part1 = 0
        alpha_part2 = 0
        if image.depth == 32:
            alpha_part1 = data.pop(0)
            alpha_part2 = data.pop(0)
        pixel_indices = data.pop(0)
        block_info = data.pop(0)
        bc1 = [0, 0, 0]
        bc2 = [0, 0, 0]
        if block_info & 2 == 0:
            bc1[0] = block_info >> 28 & 15
            bc1[1] = block_info >> 20 & 15
            bc1[2] = block_info >> 12 & 15
            bc1 = [(x << 4) + x for x in bc1]
            bc2[0] = block_info >> 24 & 15
            bc2[1] = block_info >> 16 & 15
            bc2[2] = block_info >> 8 & 15
            bc2 = [(x << 4) + x for x in bc2]
        else:
            bc1[0] = block_info >> 27 & 31
            bc1[1] = block_info >> 19 & 31
            bc1[2] = block_info >> 11 & 31
            bc2[0] = block_info >> 24 & 7
            bc2[1] = block_info >> 16 & 7
            bc2[2] = block_info >> 8 & 7
            bc2 = [x + ((y > 3) and (y - 8) or y) for x, y in zip(bc1, bc2)]
            bc1 = [(x << 3) + (x >> 2) for x in bc1]
            bc2 = [(x << 3) + (x >> 2) for x in bc2]
        flip = block_info & 1
        tcw1 = block_info >> 5 & 7
        tcw2 = block_info >> 2 & 7
        for i in range(16):
            mi = ((pixel_indices >> i) & 1) + ((pixel_indices >> (i + 15)) & 2)
            c = None
            if flip == 0 and i < 8 or flip != 0 and (i // 2 % 2) == 0:
                m = modifier_tables[tcw1][mi]
                c = [max(0, min(255, x + m)) for x in bc1]
            else:
                m = modifier_tables[tcw2][mi]
                c = [max(0, min(255, x + m)) for x in bc2]
            offset = block_index % 4
            x = (block_index - offset) % (image.size[0] // 2) * 2
            y = (block_index - offset) // (image.size[0] // 2) * 8
            if offset & 1:
                x += 4
            if offset & 2:
                y += 4
            x += i // 4
            y += i % 4
            offset = (x + (image.size[1] - y - 1) * image.size[0]) * 4
            image_pixels[offset] = c[0] / 255
            image_pixels[offset+1] = c[1] / 255
            image_pixels[offset+2] = c[2] / 255
        block_index += 1
    image.pixels = image_pixels
    image.update()
    image.pack(True)


def load_tex(filename, name):
    tex = open(filename, 'rb')
    tex_header = struct.unpack('4s3I', tex.read(16))
    constant = tex_header[1] & 0xfff
    #unknown1 = (tex_header[1] >> 12) & 0xfff
    size_shift = (tex_header[1] >> 24) & 0xf
    #unknown2 = (tex_header[1] >> 28) & 0xf
    mipmap_count = tex_header[2] & 0x3f
    width = (tex_header[2] >> 6) & 0x1fff
    height = (tex_header[2] >> 19) & 0x1fff
    #unknown3 = tex_header[3] & 0xff
    pixel_type = (tex_header[3] >> 8) & 0xff
    #unknown5 = (tex_header[3] >> 16) & 0x1fff
    offsets = array.array('I', tex.read(4 * mipmap_count))
    if pixel_type == 11:
        image = bpy.data.images.new('texture', width, height)
        decode_etc1(image, tex.read(width*height//2))
    elif pixel_type == 12:
        image = bpy.data.images.new('texture', width, height, True)
        decode_etc1(image, tex.read(width*height))
    tex.close()


def load_mrl():
    pass


def parse_vertex(raw_vertex):
    vertex = array.array('f', raw_vertex[:12])
    y=vertex[1]
    z=vertex[2]
    vertex[1]=-z
    vertex[2]=y   
    uv = array.array('f', raw_vertex[16:24])
    n  = array.array('b',raw_vertex[12:16])
    normal = [n[0]/127.0,-n[2]/127.0,n[1]/127.0]    
    t  = array.array('b',raw_vertex[28:31])
    tangent = [t[0]/127.0,-t[2]/127.0,t[1]/127.0]
    bones = list(raw_vertex[24:26] + raw_vertex[32:34])
    weights = [x / 255 for x in raw_vertex[26:28] + raw_vertex[34:36]]

    return vertex, uv, bones, weights, normal, tangent

def build_vertex(vertex, UV, NTB , bonemap, vcolor):
    groupWeights = {}
    x = vertex.co[0]
    z = -vertex.co[1]
    y = vertex.co[2]
    for vg in vertex.groups:
        if vg.weight > 0.0:
            groupWeights[vg.group] = vg.weight
    # Normalize
    totalWeight = 0
    for k in groupWeights.keys():
        totalWeight += groupWeights[k]
    if totalWeight > 0:
        normalizer = 1/totalWeight
        for gw in groupWeights:
            groupWeights[gw] *= normalizer
    #Sort Weights
    sortedWeights = sorted(groupWeights.items(),key=operator.itemgetter(1),reverse=True)
    #TruncateWeights
    weightCnt = len(sortedWeights)
    if weightCnt == 0: # no bone weight found, add a dummy bone
        sortedWeights.append([1,0])
    while weightCnt < 4:
        sortedWeights.append([sortedWeights[0][0],0])
        weightCnt += 1 
    
    raw_vertex = bytearray(36)
    # flip y and z                       
    x=vertex.co[0]
    z=-vertex.co[1]
    y=vertex.co[2]
    bm_idx = [ bonemap.index(i) if i > 0 else 0 for (i,w) in sortedWeights  ]
    raw_vertex[0:12]=struct.pack("3f",x,y,z)
    normal = NTB[0]
    tangent = NTB[1]
    flip = NTB[2]
    raw_vertex[12:16]=struct.pack("bbbb",round(normal[0]*127),round(normal[2]*127),round(-normal[1]*127),0x7F)
    raw_vertex[28:32]=struct.pack("bbbb",round(tangent[0]*127),round(tangent[2]*127),round(-tangent[1]*127),round(flip*127))


    raw_vertex[16:24]=struct.pack("2f",UV[0],UV[1]);
    raw_vertex[24:26]=struct.pack("2B",bm_idx[0], bm_idx[1]);
    raw_vertex[32:34]=struct.pack("2B", bm_idx[2],bm_idx[3])  
    raw_vertex[26:28]=struct.pack("2B",round(sortedWeights[0][1]*255.0),round(sortedWeights[1][1]*255.0))
    raw_vertex[34:36]=struct.pack("2B",round(sortedWeights[2][1]*255.0),round(sortedWeights[3][1]*255.0))
    return raw_vertex

def parse_faces(vertex_start_index, raw_faces):
    raw_faces = array.array('H', raw_faces)
    reverse = True
    faces = []
    f1 = raw_faces.pop(0)
    f2 = raw_faces.pop(0)
    while len(raw_faces) > 0:
        f3 = raw_faces.pop(0)
        if f3 == 0xffff:
            f1 = raw_faces.pop(0)
            f2 = raw_faces.pop(0)
            reverse = True
        else:
            reverse = not reverse
            if f1!=f2 and f2!=f3 and f3!=f1:
                if reverse:
                    faces.append([f1-vertex_start_index, f3-vertex_start_index, f2-vertex_start_index])
                else:
                    faces.append([f1-vertex_start_index, f2-vertex_start_index, f3-vertex_start_index])
            f1 = f2
            f2 = f3
    return faces


def build_uv_map(b_mesh, uvs, faces,tangents):
    #b_mesh.uv_textures.new()
    bm = bmesh.new()
    bm.from_mesh(b_mesh)
    uv_layer = bm.loops.layers.uv.new("UVMap")
    #for i,loop in enumerate(b_mesh.loops):
    #    b_mesh.uv_layers[0].data[i].uv = uvs[loop.vertex_index]
    for face in bm.faces:
        for loop in face.loops:
            loop[uv_layer].uv = uvs[loop.vert.index]
    color_layer = bm.verts.layers.float_vector.new("Color")
    for i,v in enumerate(bm.verts):
        v[color_layer] = tangents[i]
    bm.to_mesh(b_mesh)
    bm.free()
    b_mesh.update()

def create_skeleton( Bones, ParentIndices, BoneMatrices):
    armatureName = "Armature"
    print ("new amatureNmae", armatureName)
    armature = bpy.data.armatures.new(armatureName)
    obj = bpy.data.objects.new("skeleton", armature)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    print("Create Skeleton: Creating Bones...")
    for i,b in enumerate(Bones):
        bone = armature.edit_bones.new(b)
        if ParentIndices[i] != 255:
            bone.parent = armature.edit_bones[ParentIndices[i]]
        # print(bone.parent)
        bone.tail = mathutils.Vector([0,0.0,0.1])
        bone.matrix = BoneMatrices[i]

    print("Create Skeleton: Swaping Bones Z and Y axis")
    for b in armature.edit_bones:
        zaxis = b.z_axis
        length = 0.1 #default bone length

        if len(b.children) == 1:
            if b.children[0].head != mathutils.Vector([0.0,0.0,0.0]):
                b.tail = b.children[0].head # connect bone to child

    bpy.ops.object.mode_set(mode='OBJECT')
    return obj

def load_mod(filename, context):
    print ("import", filename)
    mod = open(filename, 'rb')
    mod_header = struct.unpack('4s4H13I', mod.read(64))
    if mod_header[0] != b'MOD\x00' or mod_header[1] != 0xe5:
        print ("wrong header")
        mod.close()
        return    
    bone_cnt = mod_header[2]
    vert_size = mod_header[8]
    long05= mod_header[9]
    long06= mod_header[10]
    UsedBoneCount = mod_header[11]
    bone_offset = mod_header[12]
    matCnt = mod_header[13]

    MatOff = mod_header[14]
    MeshOff = mod_header[15]
    vertOffset = mod_header[16]
    faceOffset = mod_header[17]

    mod.seek(bone_offset,0)
    bones=[]
    bone_parent=[]
    # read skeleteon
    for i in range(bone_cnt):
        bone_info =  struct.unpack('4B5f', mod.read(24))
        bones.append("b_"+str(i))
        bone_parent.append(bone_info[1])
    print ("Bone end")
    for a in range (bone_cnt):
        get_pos = mod.tell() + 64
        mod.seek( get_pos,0)
    bone_matrix = []
    # get bone transformation/position
    for a in range(bone_cnt):
        c = struct.unpack('16f', mod.read(64))
        # swap y and z translation
        tfm = mathutils.Matrix([(c[0],c[4],c[8],c[12]),\
                                (c[2],c[5],c[9],-c[14]),\
                                    (c[3],c[6],c[10],c[13]),\
                                        (c[4],c[7],c[11],c[15])])
        bone_matrix.append(tfm.inverted())
        print (tfm)
    print ("bone_cnt",bone_cnt,"matrix size",len(bone_matrix),bones)
    print ("parent ", bone_parent)    
    armature = create_skeleton(bones,bone_parent, bone_matrix)

    mod.seek(0x100,1)
    Pos = mod.tell()

    BoneMapArray=[]
    # read all bone map, should just be one for Merce3D
    for uc in range(UsedBoneCount):
        BoneMapArray00=[]
        UBoneCnt=1
        for i in range(UBoneCnt):
            BoneMapCount00=struct.unpack('I',mod.read(4))[0]
            for j in range(BoneMapCount00):
                BMap00=struct.unpack('B',mod.read(1))[0]
                BoneMapArray00.append(BMap00)
            Stride00=(24-BoneMapCount00)
            mod.seek(Stride00, 1)
        BoneMapArray.append(BoneMapArray00)
    print ("BoneMapArray",BoneMapArray)  

    for i in range(mod_header[3]):
        mod.seek(mod_header[15] + i * 48)
        # 0 - unk, 1 - vertex count, 2 - unk, 3 - unk  4 - vertex size, 5 - vtype,  
        # 6 - vstart, 7 - unk, 8 - unk, 9 - facePos, 10 = face_cnt, 
        # 11= Null , 12=NullB, 13 - MeshId, 14 - UBN

        mesh_info = struct.unpack('HHIHBB6I3B', mod.read(39))
        mod.seek(mod_header[16] + mesh_info[6] * mesh_info[4] + mesh_info[7])
        vertices = []
        uvs = []
        boneIdxWeight = []
        normals = []
        tangents = []
        print ("vertex size",mesh_info[4])

        for j in range(mesh_info[1]):
            vertex, uv, boneindices, weights, normal, tangent = parse_vertex(mod.read(mesh_info[4]))
            normals.append(normal)
            vertices.append(vertex)
            tangents.append(tangent)
            if len(uv) != 0:
                uvs.append(uv)
            if len (boneindices) == 0:
                print ("error, no bone indices")
            else:
                boneIdxWeight.append(list(zip(boneindices,weights)))
       
        mod.seek(mod_header[17] + mesh_info[9] * 2)
        faces = parse_faces(mesh_info[6], mod.read(mesh_info[10] * 2 + 2))
        b_mesh = bpy.data.meshes.new('mesh_{}'.format(i))
        b_object = bpy.data.objects.new('Mesh_{}'.format(i), b_mesh)
        b_mesh.from_pydata(vertices, [], faces)
        b_mesh.update(calc_edges=True)
        b_mesh.use_auto_smooth = True
        b_mesh.normals_split_custom_set_from_vertices(normals)
        #bpy.context.scene.objects.link(b_object)
        bpy.context.scene.collection.objects.link(b_object)
        if len(uvs) != 0:
            build_uv_map(b_mesh, uvs, faces, tangents)

        # add vertex groups to mesh           
        for bone in bones:
            b_object.vertex_groups.new(name=bone)
        # parent object to armature
        b_object.modifiers.new(name='Skeleton', type='ARMATURE')
        b_object.modifiers['Skeleton'].object = armature
        b_object.parent = armature 
        
        MeshId = mesh_info[13]
        BoneMap = BoneMapArray[MeshId]
        bonemap_len = len(BoneMap)

        vertex_cnt = mesh_info[1]
        # set vertex bone weights for mapped bones
        for v in range(vertex_cnt):     
               
            for bi,bw in boneIdxWeight[v]:            
                if bw > 0.0:
                    if bi >= bonemap_len:
                        print ("mesh",i,"v",v,"total",vertex_cnt,"boneIdxWeight",boneIdxWeight[v]) 
                    bone_name = "b_" + str(BoneMap[bi])                
                    b_object.vertex_groups[bone_name].add([v],bw ,'ADD')
        b_mesh.update()
    mod.close()

def CopyBytes(f,nf,bytes):
    for i in range(bytes):
        nf.write(f.read(1))
    return

def do_export(self, context, props, filepath):
    #print("Exporting to", filepath)
    selection = context.selected_objects        
    selcnt= len (selection)
    f = open(filepath,'rb')
    if selcnt == 0: return False
    if f == None: return False

    #Type = struct.unpack('4s',f.read(4))[0]
    header_bytes=f.read(64)
    mod_header = struct.unpack('4s4H13I',header_bytes)
    nf = open(filepath +".NewMOD",'wb')
    boneCnt = mod_header[2]
    meshCnt = mod_header[3]
    UsedBoneCount = mod_header[11]
    boneOff = mod_header[12]    
    MatOff = mod_header[14]
    meshOff = mod_header[15]
    vertOff = mod_header[16]
    faceOff = mod_header[17]

    # copy header until end of meshInfo section
    nf.write(header_bytes)

    CopyBytes(f,nf,vertOff-64)

    # read original bonemap
    f.seek(boneOff + boneCnt * (24 +64+64)+ 0x100)
    BoneMapArray=[]
    # read all bone map, should just be one for Merce3D
    for uc in range(UsedBoneCount):
        BoneMapArray00=[]
        UBoneCnt=1
        for i in range(UBoneCnt):
            BoneMapCount00=struct.unpack('I',f.read(4))[0]
            for j in range(BoneMapCount00):
                BMap00=struct.unpack('B',f.read(1))[0]
                BoneMapArray00.append(BMap00)
            Stride00=(24-BoneMapCount00)
            f.seek(Stride00, 1)
        BoneMapArray.append(BoneMapArray00)
    print ("BoneMapArray",BoneMapArray)     

    selection = context.selected_objects
    active_mesh = {}
    for i in range(selcnt):
        sName = re.split('_',selection[i].name)
        if sName[0] == 'Mesh':
            active_mesh[int(sName[1])]=selection[i]
    

    vertStart= [0] * meshCnt
    vertCnt =  [0] * meshCnt
    faceStart= [0] * meshCnt
    idxCnt =   [0] * meshCnt
    allfaces = []
    vertex_start_index = 0
    face_start_index = 0

    f.seek(vertOff)
    for m in range(meshCnt):
        vertStart[m]=vertex_start_index
        faceStart[m]=face_start_index
        if m in active_mesh:  # this mesh for keeping
            obj = active_mesh[m]
            editedMesh = obj.data
            UVs = [(0.0, 0.0)] * len(editedMesh.vertices)
            bm = bmesh.new()
            bm.from_mesh(editedMesh)
            bm.faces.ensure_lookup_table()
            # Get UVs
            uvIndex = 0
            for bface in bm.faces:
                for loop in bface.loops:
                    u = loop[bm.loops.layers.uv[uvIndex]].uv[0]
                    v = loop[bm.loops.layers.uv[uvIndex]].uv[1]
                    UVs[loop.vert.index] = [u,v]

            # Write Normals Stream
            NTB = [((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.0)] * len(editedMesh.vertices)  # Normal Tangent Bi-tangent
            # Get Normals
            editedMesh.loops.data.calc_tangents()
            for l in editedMesh.loops:
                NTB[l.vertex_index] = (l.normal, l.tangent, l.bitangent_sign)

            # Get face tri strip
            tri_strip = triangles_list_to_triangles_strip(editedMesh)

            # combine all MESH tri strip            
            tri_strip_len =  len(tri_strip) 
            for idx in tri_strip:
                allfaces.append(vertex_start_index + idx)    


            bm.verts.ensure_lookup_table()
            color_layer = bm.verts.layers.float_vector["Color"]
            # write vertex , assuming only using first bonemap            
            for i,v in enumerate(editedMesh.vertices):
                raw_vertex = build_vertex(v,UVs[i],NTB[i],BoneMapArray[0],bm.verts[i][color_layer])
                nf.write(raw_vertex)

            bm.to_mesh(editedMesh)
            bm.free()
            editedMesh.update()           
            mesh_vertex_cnt = len(editedMesh.vertices)
            vertCnt[m] = mesh_vertex_cnt
            idxCnt[m] = tri_strip_len
        else:  # reduce mesh to 1 invisible triangle
            # write a degenerated triangle, 3 vertex at 0,0,0
            mesh_vertex_cnt = 3
            vertCnt[m] = mesh_vertex_cnt
            nf.write(bytearray(36))
            nf.write(bytearray(36))
            nf.write(bytearray(36))            
            tri_strip_len = 4
            idxCnt[m] = tri_strip_len
            allfaces.extend([vertex_start_index,\
                                        vertex_start_index+1,\
                                        vertex_start_index+2,\
                                        vertex_start_index+2])

        vertex_start_index += mesh_vertex_cnt
        face_start_index +=  tri_strip_len
    
    
    newFaceOffset = nf.tell()
    
    # write all tri strip index 
    for idx in allfaces:
        nf.write(struct.pack('H',idx))
    
    # fix up mesh info with new vertex and face offset value
    for m in range(meshCnt):
        nf.seek(mod_header[15] + m * 48)
        nf.seek(2, 1)        
        nf.write(struct.pack('H',vertCnt[m]))
        nf.seek(6, 1)
        nf.write(struct.pack('B',36)) # force vertex struct size to 36
        nf.seek(1, 1)
        nf.write(struct.pack('I',vertStart[m]))
        nf.write(struct.pack('I',0))  # force all vertex to same format, no vertex base change
        nf.seek(4, 1)
        nf.write(struct.pack('I',faceStart[m]))
        nf.write(struct.pack('I',idxCnt[m]))

    # fix up vertex and face starting offset
    nf.seek(0xc)
    # total verex count
    nf.write(struct.pack('I',vertex_start_index)) # total vertex cnt
    nf.write(struct.pack('I',face_start_index)) # total face index cnt
    nf.seek(4,1)
    nf.write(struct.pack('I',(newFaceOffset-vertOff))) # vertex data size
    nf.seek(0x3c)
    nf.write(struct.pack('I',newFaceOffset))
    
    f.close()
    nf.close()     
    return True
    

class IMPORT_OT_mod(bpy.types.Operator):
    bl_idname = "import_scene.mod"
    bl_label = "Import MOD"
    bl_description = "Import a Mercenaries 3D model"
    bl_options = {'REGISTER', 'UNDO'}

    #filepath = bpy.props.StringProperty(name="File Path", description="Filepath used for importing the MOD file", maxlen=1024, default="")
    filepath: StringProperty(
            name="input file",
            subtype='FILE_PATH'
            )

    filename_ext = ".mod"
    def execute(self, context):
        load_mod(self.filepath, context)
        #load_tex(self.filepath.replace('.58A15856', '_BM.241F5DEB'), 'test')
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}

            
###### EXPORT OPERATOR #######
class EXPORT_OT_mod(bpy.types.Operator):
    """Export Selected Meshes to MOD file"""
    bl_idname = "export_mesh.mod"
    bl_label = "Export Mercenaries 3D model"

    filename_ext = ".mod"

    #@classmethod
    #def poll(cls, context):
    #    return context.active_object.type in {'MESH', 'CURVE', 'SURFACE', 'FONT'}
    filepath: StringProperty(
            name="origin mod file",
            subtype='FILE_PATH'
            )


    def execute(self, context):
        props = self.properties
        filepath = bpy.path.ensure_ext(self.filepath, self.filename_ext)
        exported = do_export(self, context, props, filepath)

        if exported:
            print(filepath)

        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager

        if True:
            # File selector
            wm.fileselect_add(self) # will run self.execute()
            return {'RUNNING_MODAL'}
        
def menu_func(self, context):
    self.layout.operator(IMPORT_OT_mod.bl_idname, text="Mercenaries 3D Model (.mod)")

def export_menu_func(self, context):
    self.layout.operator(EXPORT_OT_mod.bl_idname, text="Mercenaries 3D Model (.mod)")


def register():
    #bpy.utils.register_module(__name__)
    bpy.utils.register_class(IMPORT_OT_mod)
    bpy.utils.register_class(EXPORT_OT_mod)
    bpy.types.TOPBAR_MT_file_import.append(menu_func)
    bpy.types.TOPBAR_MT_file_export.append(export_menu_func)

def unregister():
    #bpy.utils.unregister_module(__name__)
    bpy.utils.unregister_class(IMPORT_OT_mod)
    bpy.utils.unregister_class(EXPORT_OT_mod)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func)
    bpy.types.TOPBAR_MT_file_export.remove(export_menu_func)


if __name__ == "__main__":
    register()
