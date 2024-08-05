import torch
import torch.nn as nn
from scene.hexplane import HexPlaneField
import torch.nn.init as init



class TransformerLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        att_output, _ = self.attention(x, x, x)
        att_output = self.dropout(att_output)
        # Residual connection and layer normalization
        x = self.norm1(x + att_output)
        
        # MLP
        mlp_output = self.linear2(nn.ReLU()(self.linear1(x)))
        mlp_output = self.dropout(mlp_output)
        # Residual connection and layer normalization
        x = self.norm2(x + mlp_output)
        
        return x

class Deformation(nn.Module):
    def __init__(self, D=6, W=256, input_ch=27, input_ch_time=9, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.no_grid = args.no_grid
        
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = self.create_net()
        self.args = args
        
    def create_net(self):
        mlp_out_dim = 0
        if self.no_grid:
            self.attention_out = [nn.Linear(4,self.W)]
        else:
            self.attention_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim ,self.W)]
        
        # transformer_layers = []
        for i in range(self.D-1):
            self.attention_out.append(TransformerLayer(self.W, self.W, num_heads=4, dropout=0.1))
        
        self.attention_out = nn.Sequential(*self.attention_out)
        
        return  \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
    
    def query_time(self, rays_pts_emb, time_emb):
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            h = grid_feature
        # print(type(h))
        h = self.attention_out(h)
        return h

    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, time_emb):
        hidden = self.query_time(rays_pts_emb, time_emb).float()
        
        if self.args.no_dx:
            pts = rays_pts_emb[:, :3]
        else:
            dx = self.pos_deform(hidden)
            pts = rays_pts_emb[:, :3] + dx
        
        if self.args.no_ds:
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = scales_emb[:,:3] + ds
            
        if self.args.no_dr:
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)
            rotations = rotations_emb[:,:4] + dr
            
        if self.args.no_do:
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
            opacity = opacity_emb[:,:1] + do

        return pts, scales, rotations, opacity
    
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None, time_emb=None):
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, time_emb)

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        return list(self.grid.parameters()) 

class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(),
            nn.Linear(timenet_width, timenet_output))
        
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(4+3)+((4+3)*scale_rotation_pe)*2, input_ch_time=timenet_output, args=args)
        
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
    
    def forward(self, point, scales=None, rotations=None, opacity=None, times_sel=None):
        if times_sel is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, times_sel)
        else:
            return self.forward_static(point)
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points

    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        means3D, scales, rotations, opacity = self.deformation_net( point,
                                                scales,
                                                rotations,
                                                opacity,
                                                times_sel)
        return means3D, scales, rotations, opacity
    
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
