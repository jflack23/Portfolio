close all

%Read in Posterior Anterior IMU of left and right leg
Xl=Xl_e12;
Xr=Xr_e12;

%Declare latency & Percentage of the stride to be perturbed
lat=36;
percent=15;

%Initialising Variables and Constants
time=[0:1/512*1000:length(Xl)/512*1000-1000/512];
i=0;
Datal=[];
Datar=[];

peakl=[];
ICl=[];
TOl=[];
negl=[];
countpl=0;
counttl=0;
countnl=0;
kl=100;
peaked=0;
nl=0;
swingl=1;
init_l=[];
end_l=[];

peakr=[];
ICr=[];
TOr=[];
negr=[];
countpr=0;
counttr=0;
countnr=0;
kr=100;
peaked=0;
nr=0;
swingr=1;
init_r=[];
end_r=[];

l=0;

%Signal Coming In
for i=1:length(Xl)
    if i>5120
        %Finding mean and SD of PA in left leg for first 10
        %seconds of signal
        Datal=Xl(1:i);
        Data_F=lowpass(Datal,60,512);
        pd = fitdist(Data_F,'Normal');
        mu_l=pd.mu;
        sig_l=pd.sigma;
        
        %Calculating stride time of left leg
        [b,p]=findpeaks(Datal,1,'MinPeakDistance',256,'MinPeakProminence',5);
        stride_l=p(length(p))-p(length(p)-1);
        peakl(countpl+1)=p(length(p));
        
        
        %Finding mean and SD of PA in right leg for first 10
        %seconds of signal
        Datar=Xr(1:i);
        Data_F=lowpass(Datar,60,512);
        pd = fitdist(Data_F,'Normal');
        mu_r=pd.mu;
        sig_r=pd.sigma;
        
        %Calculating stride time of right leg
        [b,p]=findpeaks(Datar,1,'MinPeakDistance',256,'MinPeakProminence',5);
        stride_r=p(length(p))-p(length(p)-1);
        peakr(countpr+1)=p(length(p));
        break
    end
end



for j=i:length(Xl)
    
    %Read in last 250 datapoints
    Datal=Xl(j-250:j);
    Datar=Xr(j-250:j);
    
    %Major Peak Identification left leg
    if Datal(249)>=Datal(248) && Datal(249)>=Datal(250) && Datal(249)>2*sig_l+mu_l && swingr==1 && kl>250 && peaked==0 && min(Datal(220:250))<-0.3*mu_l
        
        %Saving peak position
        countpl=countpl+1;
        peakl(countpl+1)=j-2;
        
        %Stating that the left leg has peaked
        peaked=1;
        
        %Finding the next minimum in the last 0.25s
        [v,l]=min(Datal(125:250));
        negl(countpl)=j+l-127;
        
        nl=0; %Number of peaks of left leg since peak
        kl=0; %Number of frames since left leg peak
        I=0; %IC found set to 0
        
        a=l+125; %Setting a to be the minimum value found
        swingr=0; %Resetting right leg swing counter
        
        init_r(countpl)=round(j+lat/1000*512); %Calculating the start time for pertubation of right leg using latency value
        
        if peakl(countpl+1)<peakl(countpl)+512
            stride_l=peakl(countpl+1)-peakl(countpl); %Calculating th previous stride length of left leg
        end
        
        %Calculating the end point of pertubation of right leg
        end_r(countpl)=round(init_r(countpl)+percent/100*stride_r);
        
        %Calculating initial contact by travelling back through the data to find
        %the closest peak above the mean
        while I==0
           if Datal(a-1)>=Datal(a-2) && Datal(a-1)>=Datal(a) && Datal(a-1)>mu_l
               counttl=counttl+1;
               ICl(counttl)=j-252+a;
               I=1;
           else
               a=a-1;
           end
        end
    end
    
    %Major Peak Identification right leg
    if Datar(249)>=Datar(248) && Datar(249)>=Datar(250) && Datar(249)>2*sig_r+mu_r && swingl==1 && kr>250 && peaked==0 && min(Datar(220:250))<-0.3*mu_r
        
        %Saving peak position
        countpr=countpr+1;
        peakr(countpr+1)=j-2;
        
        %Stating that the right leg has peaked
        peaked=1;
        
        %Finding the next minimum n the last 0.25s
        [v,l]=min(Datar(125:250));
        negr(countpr)=j+l-127;
        
        nr=0; %Number of peaks of right leg since peak
        kr=0; %Number of framessince left leg peak
        I=0; %IC found set to 0
        
        a=l+125; %Setting a to be the minimum value found
        swingl=0; %Resetting leftleg swing counter
        
        init_l(countpr)=round(j+lat/1000*512); %Calculating the start time for pertubation of left leg using latency value
       
        if peakr(countpr+1)<peakr(countpr)+512
            stride_r=peakr(countpr+1)-peakr(countpr); %Calculating th previous stride length of left leg
        end
        %Calculating the end point of pertubation of right leg
        end_l(countpr)=round(init_l(countpr)+percent/100*stride_l);
       
        %Calculating initial contact by travelling back through the data to find
        %the closest peak above the mean
        while I==0
           if Datar(a-1)>=Datar(a-2) && Datar(a-1)>=Datar(a) && Datar(a-1)>mu_r
               counttr=counttr+1;
               ICr(counttr)=j-252+a;
               I=1;
           else
               a=a-1;
           end
        end
    end
    
    %Toe off for Left
    if Datal(249)<=Datal(248) && Datal(249)<=Datal(250) && Datal(249)<-mu_l && nl<1 && kl>50 && peaked==1
        countnl=countnl+1; 
        TOl(countnl)=j-2; %Saving value of TO
        nl=nl+1; %Increasing number of peaks found in left
        peaked=0; %Setting peaked to 0
        swingl=1; %Stating that the left leg is in swing
        swingr=0;
    end
    
    %Toe off for Right
    if Datar(249)<=Datar(248) && Datar(249)<Datar(250) && Datar(249)<-mu_r && nr<1 && kr>50 && peaked==1
        countnr=countnr+1;
        TOr(countnr)=j-2; %Saving value of TO
        nr=nr+1; %Increaseing number of peaks found in right
        peaked=0; %Setting peaked to 0
        swingr=1; %Stating that the right leg is in swing
        swingl=0;
    end
    
    %Increasing k values
    kl=kl+1;
    kr=kr+1;
end
