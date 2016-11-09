/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package prediction.apparel.views;

import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.ButtonGroup;
import javax.swing.JOptionPane;
import javax.swing.JRadioButton;
import prediction.apparel.databaseConn.DBAccessJava;

/**
 *
 * @author Lakini
 */
public class ManualPredictionUI extends javax.swing.JFrame {
    
    JRadioButton optionRF = null;
    JRadioButton optionLR = null;
    JRadioButton optionSVM = null;
    JRadioButton optionKNN = null;
    JRadioButton optionDT = null;
    FileWriter fileWriter = null;

    /**
     * Creates new form ManualPredictionUI
     */
    public ManualPredictionUI() throws IOException {
        initComponents();
        
        
        optionRF = new JRadioButton("Random Forest");
        optionLR = new JRadioButton("Logistic Regression");
        optionSVM = new JRadioButton("Support Vector Machine");
        optionKNN = new JRadioButton("K Nearest Neighbours");
        optionDT = new JRadioButton("Decision Trees");
 
        ButtonGroup group = new ButtonGroup();
        group.add(optionRF);
        group.add(optionLR);
        group.add(optionSVM);
        group.add(optionKNN);
        group.add(optionDT);
        
        setLayout(new FlowLayout());
 
        add(optionRF);
        add(optionLR);
        add(optionSVM);
        add(optionKNN);
        add(optionDT);
 
        pack();
        
        ////////////////////////////
        fileWriter = new FileWriter("src/prediction/apparel/csv/modelManual.csv");
        fileWriter.write("modelName");
        fileWriter.write("\n");

	
            
        /////////////////////////
        
        RadioButtonActionListener actionListener = new RadioButtonActionListener();
	optionRF.addActionListener(actionListener);
        optionLR.addActionListener(actionListener);
	optionSVM.addActionListener(actionListener);
        optionKNN.addActionListener(actionListener);
        optionDT.addActionListener(actionListener);
        
       
        
    }
    
        class RadioButtonActionListener implements ActionListener {

		@Override
		public void actionPerformed(ActionEvent event) {
                    System.out.println("Inside radio button");
			JRadioButton button = (JRadioButton) event.getSource();
			if (button == optionRF) {
                            try {
                                fileWriter.append("RandomForest");
                                fileWriter.flush();
                                fileWriter.close();
                            } catch (IOException ex) {
                                Logger.getLogger(ManualPredictionUI.class.getName()).log(Level.SEVERE, null, ex);
                            }


			} else if (button == optionLR) {
                            try {
                                fileWriter.append("LogisticReg");
                                fileWriter.flush();
                                fileWriter.close();
                            } catch (IOException ex) {
                                Logger.getLogger(ManualPredictionUI.class.getName()).log(Level.SEVERE, null, ex);
                            }

			} else if (button == optionSVM) {
                            try {
                                fileWriter.append("SVMC");
                                fileWriter.flush();
                                fileWriter.close();
                            } catch (IOException ex) {
                                Logger.getLogger(ManualPredictionUI.class.getName()).log(Level.SEVERE, null, ex);
                            }

			}else if (button == optionDT) {
                            try {
                                fileWriter.append("DecisionTree");
                            } catch (IOException ex) {
                                Logger.getLogger(ManualPredictionUI.class.getName()).log(Level.SEVERE, null, ex);
                            }

			}else if (button == optionKNN) {
                            try {
                                fileWriter.append("kNN9");
                                fileWriter.flush();
                                fileWriter.close();
                            } catch (IOException ex) {
                                Logger.getLogger(ManualPredictionUI.class.getName()).log(Level.SEVERE, null, ex);
                            }

			}
                        
		}        
	}
    
    
    
    
    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        btnManualPrediction = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        btnManualPrediction.setText("Predict");
        btnManualPrediction.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnManualPredictionActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(157, 157, 157)
                .addComponent(btnManualPrediction, javax.swing.GroupLayout.PREFERRED_SIZE, 99, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(144, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(235, 235, 235)
                .addComponent(btnManualPrediction, javax.swing.GroupLayout.PREFERRED_SIZE, 33, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(32, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void btnManualPredictionActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnManualPredictionActionPerformed
        System.out.println("Insode Button");
        DBAccessJava dbaccess = DBAccessJava.getDbCon();
        try {
            System.out.println("goint to call pythin code");
            Process p = Runtime.getRuntime().
                    exec("python src/prediction/apparel/pythoncode/trainBestModelManual.py");
            
            BufferedReader stdInput = new BufferedReader(new
                                         InputStreamReader(p.getInputStream()));
            
 
            BufferedReader stdError = new BufferedReader(new
                                         InputStreamReader(p.getErrorStream()));
            
            String s = null;
            
            System.out.println("Here is the standard output of the command:\n");
            while ((s = stdInput.readLine()) != null) {
                System.out.println(s);
            }
             
            // read any errors from the attempted command
            System.out.println("Here is the standard error of the command (if any):\n");
            while ((s = stdError.readLine()) != null) {
                System.out.println(s);
            }

            JOptionPane.showMessageDialog(null,"Sucessfully Predicted!!","Message",JOptionPane.OK_OPTION);
                      
            dbaccess.insert(" LOAD DATA INFILE 'C:/Users/Lakini/Documents/CDAP-mypart/ApparelPrediction/src/prediction/apparel/csv/predictedchurn.csv' INTO TABLE predicted_values FIELDS TERMINATED BY ' ' (ID, Name, Churn);");

            //dispose();
            
        } catch (IOException ex) {
            Logger.getLogger(MainUI.class.getName()).log(Level.SEVERE, null, ex);
        }     catch (SQLException ex) {
            Logger.getLogger(ManualPredictionUI.class.getName()).log(Level.SEVERE, null, ex);
        }
        new MainUI().setVisible(true);
        dispose();
        
        
        
        
        
        
    }//GEN-LAST:event_btnManualPredictionActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(ManualPredictionUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(ManualPredictionUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(ManualPredictionUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(ManualPredictionUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                try {
                    new ManualPredictionUI().setVisible(true);
                } catch (IOException ex) {
                    Logger.getLogger(ManualPredictionUI.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton btnManualPrediction;
    // End of variables declaration//GEN-END:variables
}
