function f = resultmat_sparse(r11,r12,t11,t12,b1,theta_x,theta_gx)

% mean bias, std, median bias, interquartile range
mean11=mean(r11); std11=std(r11); median11=median(r11);  iqr11=iqr(r11);
mean12=mean(r12); std12=std(r12); median12=median(r12);  iqr12=iqr(r12);

mat11=[(mean11(1)-b1)    (mean11(4)-b1)     (mean11(7)-b1)     (mean11(10)-b1)    (mean11(13)-b1) ;  % beta
       std11(1)          std11(4)           std11(7)           std11(10)          std11(13)       ;
       t11(1,1)         t11(1,2)            t11(1,3)            t11(1,4)           t11(1,5)   ;
       iqr11(1)          iqr11(4)            iqr11(7)         iqr11(10)          iqr11(13)           ;
       
     (mean11(2)-theta_x)    (mean11(5)-theta_x)     (mean11(8)-theta_x)     (mean11(11)-theta_x)    (mean11(14)-theta_x)    ; 
     std11(2)          std11(5)           std11(8)           std11(11)          std11(14)     ;
       t11(2,1)         t11(2,2)            t11(2,3)            t11(2,4)           t11(2,5)   ;
     iqr11(2)          iqr11(5)            iqr11(8)         iqr11(11)          iqr11(14)       ;
       
      (mean11(3)-theta_gx)    (mean11(6)-theta_gx)     (mean11(9)-theta_gx)     (mean11(12)-theta_gx)    (mean11(15)-theta_gx)     ; 
      std11(3)          std11(6)           std11(9)           std11(12)          std11(15)       ;
       t11(3,1)         t11(3,2)            t11(3,3)            t11(3,4)           t11(3,5)   ;
      iqr11(3)          iqr11(6)            iqr11(9)         iqr11(12)          iqr11(15)     ];
       
mat12=[(mean12(1)-b1)    (mean12(4)-b1)     (mean12(7)-b1)     (mean12(10)-b1)    (mean12(13)-b1) ;  % beta
       std12(1)          std12(4)           std12(7)           std12(10)          std12(13)       ;
       t12(1,1)         t12(1,2)            t12(1,3)            t12(1,4)           t12(1,5)   ;
       iqr12(1)          iqr12(4)            iqr12(7)         iqr12(10)          iqr12(13)           ;
       
     (mean12(2)-theta_x)    (mean12(5)-theta_x)     (mean12(8)-theta_x)     (mean12(11)-theta_x)    (mean12(14)-theta_x)    ; 
     std12(2)          std12(5)           std12(8)           std12(11)          std12(14)     ;
       t12(2,1)         t12(2,2)            t12(2,3)            t12(2,4)           t12(2,5)   ;
     iqr12(2)          iqr12(5)            iqr12(8)         iqr12(11)          iqr12(14)       ;
       
      (mean12(3)-theta_gx)    (mean12(6)-theta_gx)     (mean12(9)-theta_gx)     (mean12(12)-theta_gx)    (mean12(15)-theta_gx)     ; 
      std12(3)          std12(6)           std12(9)           std12(12)          std12(15)       ;
       t12(3,1)         t12(3,2)            t12(3,3)            t12(3,4)           t12(3,5)   ;
      iqr12(3)          iqr12(6)            iqr12(9)         iqr12(12)          iqr12(15)     ];


mat1=[mat11 mat12];
f = mat1;
end